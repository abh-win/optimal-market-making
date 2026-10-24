#include "mm/engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mm {

QueueModel parseQueueModel(const std::string& s) {
  if (s == "risk-averse") return QueueModel::RiskAverse;
  if (s == "proportional") return QueueModel::Proportional;
  if (s == "power2") return QueueModel::PowerProb2;
  if (s == "power3") return QueueModel::PowerProb3;
  throw std::invalid_argument("queue model: risk-averse | proportional | power2 | power3");
}

const char* toString(QueueModel q) {
  switch (q) {
    case QueueModel::RiskAverse: return "risk-averse";
    case QueueModel::Proportional: return "proportional";
    case QueueModel::PowerProb2: return "power2";
    case QueueModel::PowerProb3: return "power3";
  }
  return "?";
}

Engine::Engine(const MarketData& md, const SignalParams& sp, EngineConfig cfg)
    : md_(md), cfg_(std::move(cfg)), sig_(std::make_unique<Signals>(sp)) {}

void Engine::run(Strategy& s, double t0, double t1) {
  s.reset();
  size_t ti = 0, di = 0;
  const auto& T = md_.trades;
  const auto& D = md_.depth;
  if (t0 > 0) {
    ti = std::lower_bound(T.begin(), T.end(), t0, [](const Trade& x, double t) { return x.ts < t; }) - T.begin();
    di = std::lower_bound(D.begin(), D.end(), t0, [](const Depth& x, double t) { return x.ts < t; }) - D.begin();
  }
  while (di < D.size() && D[di].ts < t1) {
    // Trades up to and including this snapshot's timestamp come first.
    while (ti < T.size() && T[ti].ts <= D[di].ts) {
      processTimers(T[ti].ts);
      onTrade(T[ti]);
      ++ti;
    }
    const Depth& d = D[di];
    processTimers(d.ts);
    onDepth(d);
    if (d.ts >= next_decision_) {
      Quote q = s.decide(*this, d);
      applyQuote(q, d.ts);
      next_decision_ = d.ts + cfg_.decision_ms;
    }
    ++di;
  }
  // Trades after the last snapshot of the window.
  while (ti < T.size() && T[ti].ts < t1) {
    processTimers(T[ti].ts);
    onTrade(T[ti]);
    ++ti;
  }
  finalizeMarkouts(1e300, st_.mid);
}

void Engine::processTimers(double now) {
  for (int side : {+1, -1}) {
    Side& sd = side > 0 ? bid_ : ask_;
    Order& o = sd.cur;
    if (o.cancel_pending && now >= o.cancel_at) o = Order{};
    if (sd.next.pending && now >= sd.next.active_at) {
      if (o.live && !o.cancel_pending) o = Order{};  // replace lands together with the cancel
      if (!o.live) {
        o = sd.next;
        sd.next = Order{};
        o.pending = false;
        o.live = true;
        double q = have_depth_ ? last_.qtyAt(side, o.price, md_.tick) : 0;
        o.queue_ahead = std::max(0.0, q);  // join the back of the visible queue
        o.traded_at_level = 0;
      }
    }
  }
}

void Engine::fill(int side, Order& o, double qty, double price, double ts) {
  qty = std::min(qty, o.qty);
  if (qty <= 0) return;
  o.qty -= qty;
  if (o.qty <= 1e-12) o = Order{};
  st_.position += side * qty;
  st_.cash -= side * qty * price;
  double fee = cfg_.maker_fee * qty * price;
  st_.cash -= fee;
  st_.fees += fee;
  spread_capture_ += side * (st_.mid - price) * qty;
  Fill f{ts, side, price, qty, st_.mid, sig_->regime(), {}};
  fills_.push_back(std::move(f));
}

void Engine::onTrade(const Trade& t) {
  sig_->onTrade(t);
  // Seller-initiated trades hit bids; buyer-initiated trades lift asks.
  Order& o = t.side < 0 ? bid_.cur : ask_.cur;
  const int side = t.side < 0 ? +1 : -1;
  if (!o.live) return;
  bool through = side > 0 ? t.price < o.price - 0.5 * md_.tick : t.price > o.price + 0.5 * md_.tick;
  if (through) {
    fill(side, o, o.qty, o.price, t.ts);
  } else if (std::fabs(t.price - o.price) < 0.5 * md_.tick) {
    o.traded_at_level += t.qty;
    double beyond = t.qty - o.queue_ahead;
    o.queue_ahead = std::max(0.0, o.queue_ahead - t.qty);
    if (beyond > 0) fill(side, o, beyond, o.price, t.ts);
  }
}

void Engine::updateQueueOnDepth(int side, Order& o, const Depth& d) {
  if (!o.live) return;
  // Opposite best crossing our price => we were filled.
  if ((side > 0 && d.ask() <= o.price + 1e-12) || (side < 0 && d.bid() >= o.price - 1e-12)) {
    fill(side, o, o.qty, o.price, d.ts);
    return;
  }
  double prev = last_.qtyAt(side, o.price, md_.tick);
  double now = d.qtyAt(side, o.price, md_.tick);
  if (now < 0) {  // level no longer visible: keep our estimate
    o.traded_at_level = 0;
    return;
  }
  // Our level is now the best (or better than the best): nobody ahead of us
  // once the visible depth at our price is gone.
  if (now == 0) o.queue_ahead = 0;
  if (prev > 0) {
    double cancels = std::max(0.0, prev - now - o.traded_at_level);
    if (cancels > 0 && o.queue_ahead > 0) {
      double x = std::clamp(o.queue_ahead / prev, 0.0, 1.0);
      double frac = 0;
      switch (cfg_.queue) {
        case QueueModel::RiskAverse: frac = 0; break;
        case QueueModel::Proportional: frac = x; break;
        case QueueModel::PowerProb2: frac = x * x / (x * x + (1 - x) * (1 - x)); break;
        case QueueModel::PowerProb3: frac = std::pow(x, 3) / (std::pow(x, 3) + std::pow(1 - x, 3)); break;
      }
      o.queue_ahead = std::max(0.0, o.queue_ahead - cancels * frac);
    }
  }
  o.queue_ahead = std::min(o.queue_ahead, now);
  o.traded_at_level = 0;
}

void Engine::onDepth(const Depth& d) {
  sig_->onDepth(d);
  st_.ts = d.ts;
  st_.mid = d.mid();
  if (have_depth_) {
    updateQueueOnDepth(+1, bid_.cur, d);
    updateQueueOnDepth(-1, ask_.cur, d);
  }
  last_ = d;
  have_depth_ = true;
  if (d.ts >= next_sample_) {
    curve_.push_back({d.ts, st_.equity()});
    pos_curve_.push_back({d.ts, st_.position});
    mid_hist_.push_back({d.ts, st_.mid});
    next_sample_ = d.ts + 1000;
    finalizeMarkouts(d.ts, st_.mid);
  }
}

void Engine::finalizeMarkouts(double ts, double mid) {
  // Fills whose horizons have elapsed get their markouts from the mid history.
  double hmax = cfg_.markout_s.empty() ? 0 : cfg_.markout_s.back() * 1000;
  while (markout_cursor_ < fills_.size() && fills_[markout_cursor_].ts + hmax <= ts) {
    Fill& f = fills_[markout_cursor_];
    for (double h : cfg_.markout_s) {
      double target = f.ts + h * 1000;
      auto it = std::lower_bound(mid_hist_.begin(), mid_hist_.end(), target,
                                 [](const std::pair<double, double>& a, double t) { return a.first < t; });
      double m = it == mid_hist_.end() ? mid : it->second;
      f.markout.push_back(f.side * (m - f.price));
    }
    ++markout_cursor_;
  }
}

void Engine::applyQuote(const Quote& q, double now) {
  const double lots = st_.position / cfg_.lot;
  auto place = [&](Side& sd, bool on, double price, int side) {
    // Inventory limit: never quote a side that could breach max_position.
    if (on && side * lots + 1 > cfg_.max_position + 1e-9) on = false;
    // Never cross the book (we are a passive maker).
    if (on && have_depth_) {
      if (side > 0 && price >= last_.ask() - 1e-12) price = last_.ask() - md_.tick;
      if (side < 0 && price <= last_.bid() + 1e-12) price = last_.bid() + md_.tick;
    }
    price = std::round(price / md_.tick) * md_.tick;
    Order& cur = sd.cur;
    bool cur_active = cur.live && !cur.cancel_pending;
    // The price we will be quoting once in-flight messages land.
    // Distance (ticks) by which the resting order is more passive than the target.
    auto passive_by = [&](double resting) { return side * (price - resting) / md_.tick; };
    bool keep_cur = cur_active && !sd.next.pending && passive_by(cur.price) > -0.5 &&
                    passive_by(cur.price) < cfg_.keep_ticks + 0.5;
    bool want_same = (sd.next.pending && std::fabs(sd.next.price - price) < 0.5 * md_.tick) || keep_cur;
    if (!on) {
      sd.next = Order{};
      if (cur.live && !cur.cancel_pending) {
        cur.cancel_pending = true;
        cur.cancel_at = now + cfg_.latency_ms;
      }
      return;
    }
    if (want_same) return;  // keep queue priority
    if (cur.live && !cur.cancel_pending) {
      cur.cancel_pending = true;
      cur.cancel_at = now + cfg_.latency_ms;
    }
    sd.next = Order{};
    sd.next.pending = true;
    sd.next.price = price;
    sd.next.qty = cfg_.lot;
    sd.next.active_at = now + cfg_.latency_ms;
    ++quote_updates_;
  };
  place(bid_, q.bid_on, q.bid, +1);
  place(ask_, q.ask_on, q.ask, -1);
}

Metrics computeMetrics(const Engine& e, double tick) {
  Metrics m;
  const auto& c = e.equityCurve();
  if (c.size() < 2) return m;
  m.pnl = c.back().second - c.front().second;
  m.hours = (c.back().first - c.front().first) / 3.6e6;
  m.pnl_per_hour = m.pnl / std::max(m.hours, 1e-9);
  // 15-minute PnL increments -> Sharpe, scaled to one day (x sqrt(96); markets trade 24/7).
  std::vector<double> inc;
  double next = c.front().first + 9e5, last = c.front().second;
  for (const auto& [t, eq] : c) {
    if (t >= next) {
      inc.push_back(eq - last);
      last = eq;
      next += 9e5;
    }
  }
  if (inc.size() >= 2) {
    double mu = 0, var = 0;
    for (double x : inc) mu += x;
    mu /= inc.size();
    for (double x : inc) var += (x - mu) * (x - mu);
    var /= (inc.size() - 1);
    m.sharpe_hourly = var > 0 ? mu / std::sqrt(var) * std::sqrt(96.0) : 0;
  }
  double peak = -1e300;
  for (const auto& [t, eq] : c) {
    peak = std::max(peak, eq);
    m.max_drawdown = std::max(m.max_drawdown, peak - eq);
  }
  m.fees = e.state().fees;
  m.spread_capture = e.spreadCapture();
  m.inventory_pnl = m.pnl - m.spread_capture + m.fees;
  m.fills = static_cast<long>(e.fills().size());
  for (const auto& f : e.fills()) m.volume += f.qty;
  const auto& pc = e.positionCurve();
  double lim = e.config().max_position * e.config().lot;
  for (const auto& [t, p] : pc) {
    m.mean_abs_position += std::fabs(p);
    m.time_at_limit += std::fabs(p) >= lim - 1e-9;
  }
  m.mean_abs_position /= std::max<size_t>(1, pc.size());
  m.time_at_limit /= std::max<size_t>(1, pc.size());
  size_t H = e.config().markout_s.size();
  m.markout_mean.assign(H, 0);
  long n = 0;
  for (const auto& f : e.fills()) {
    if (f.markout.size() != H) continue;
    for (size_t h = 0; h < H; ++h) m.markout_mean[h] += f.markout[h] / tick;
    ++n;
  }
  for (auto& x : m.markout_mean) x /= std::max(1L, n);
  m.quote_updates = e.quoteUpdates();
  return m;
}

}  // namespace mm
