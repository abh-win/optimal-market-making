#pragma once

// Tick-level limit-order-book backtester with queue-position tracking.
//
// Event loop: trades and depth snapshots are replayed in exchange-time order
// (trades stamped at or before a snapshot are applied first; the snapshot
// already reflects them). The strategy is asked for quotes on a fixed decision
// clock; order entries and cancels take effect after `latency_ms`.
//
// Queue model for a resting order at price p:
//   * on arrival it joins the back of the visible queue at p (queue_ahead = depth);
//   * a trade at p consumes queue_ahead first (FIFO); volume beyond it fills us;
//   * a trade through p (worse price), or the opposite best quote crossing p,
//     fills us completely;
//   * depth decreases at p that trades do not explain are cancellations; how
//     many of them were ahead of us is the QueueModel (risk-averse: none;
//     proportional; or the power-law probability model of hftbacktest-style
//     simulators, f(x) = x^n / (x^n + (1-x)^n) with x = queue_ahead / depth).
// Our orders are assumed small enough not to move the market.

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mm/data.h"
#include "mm/signals.h"

namespace mm {

enum class QueueModel { RiskAverse, Proportional, PowerProb2, PowerProb3 };
QueueModel parseQueueModel(const std::string& s);
const char* toString(QueueModel q);

struct EngineConfig {
  double latency_ms = 20;          // order entry / cancel latency
  double decision_ms = 100;        // strategy decision clock
  double lot = 1;                  // order size (base units)
  double max_position = 5;         // in lots; quotes that would breach are suppressed
  double maker_fee = 0.0;          // fraction of notional (negative = rebate)
  QueueModel queue = QueueModel::PowerProb2;
  // Queue-priority retention: keep a resting order instead of cancel/replace when it
  // sits on the passive side of the new target by at most this many ticks.
  int keep_ticks = 0;
  std::vector<double> markout_s = {1, 10, 60};
};

struct Quote {
  bool bid_on = false, ask_on = false;
  double bid = 0, ask = 0;  // prices
};

struct Fill {
  double ts;
  int side;      // +1 we bought (bid filled), -1 we sold
  double price;
  double qty;
  double mid;    // mid at fill time
  int regime;    // HMM regime at fill time
  std::vector<double> markout;  // per markout horizon: side * (mid(t+h) - price), price units
};

struct EngineState {
  double ts = 0;
  double position = 0;  // base units
  double cash = 0;
  double fees = 0;
  double mid = 0;
  double equity() const { return cash + position * mid; }
};

class Strategy;

class Engine {
 public:
  Engine(const MarketData& md, const SignalParams& sp, EngineConfig cfg);

  // Runs the whole data set (or [t0, t1)). Strategy is called on the decision clock.
  void run(Strategy& s, double t0 = -1, double t1 = 1e300);

  const EngineState& state() const { return st_; }
  const Signals& signals() const { return *sig_; }
  const EngineConfig& config() const { return cfg_; }
  double tick() const { return md_.tick; }
  const std::vector<Fill>& fills() const { return fills_; }
  const std::vector<std::pair<double, double>>& equityCurve() const { return curve_; }  // (ts, equity), 1/s
  const std::vector<std::pair<double, double>>& positionCurve() const { return pos_curve_; }
  double spreadCapture() const { return spread_capture_; }
  long quoteUpdates() const { return quote_updates_; }

 private:
  struct Order {
    bool live = false, pending = false, cancel_pending = false;
    double price = 0, qty = 0, queue_ahead = 0;
    double active_at = 0, cancel_at = 0;
    double traded_at_level = 0;  // trade volume at our price since the last snapshot
  };
  // Per side: the resting order and a pending replacement. A replaced order
  // stays fillable until its cancel reaches the exchange (stale-quote risk).
  struct Side {
    Order cur, next;
  };

  void onTrade(const Trade& t);
  void onDepth(const Depth& d);
  void processTimers(double now);
  void applyQuote(const Quote& q, double now);
  void fill(int side, Order& o, double qty, double price, double ts);
  void updateQueueOnDepth(int side, Order& o, const Depth& d);
  void finalizeMarkouts(double ts, double mid);

  const MarketData& md_;
  EngineConfig cfg_;
  std::unique_ptr<Signals> sig_;
  EngineState st_;
  Side bid_, ask_;
  Depth last_{};
  bool have_depth_ = false;
  double next_decision_ = 0, next_sample_ = 0;
  std::vector<Fill> fills_;
  size_t markout_cursor_ = 0;
  std::vector<std::pair<double, double>> curve_, pos_curve_, mid_hist_;
  double spread_capture_ = 0;
  long quote_updates_ = 0;
};

// Strategy interface. `decide` is called on the decision clock with the
// engine (state, signals, book) and returns the desired quotes.
class Strategy {
 public:
  virtual ~Strategy() = default;
  virtual std::string name() const = 0;
  virtual void reset() {}
  virtual Quote decide(const Engine& e, const Depth& book) = 0;
};

struct Metrics {
  double pnl = 0, pnl_per_hour = 0, sharpe_hourly = 0 /* daily Sharpe from 15-min PnL */, max_drawdown = 0;
  double fees = 0, spread_capture = 0, inventory_pnl = 0;
  long fills = 0;
  double volume = 0;
  double mean_abs_position = 0, time_at_limit = 0;
  std::vector<double> markout_mean;  // per horizon, ticks (negative = adverse selection)
  double hours = 0;
  long quote_updates = 0;
};
Metrics computeMetrics(const Engine& e, double tick);

}  // namespace mm
