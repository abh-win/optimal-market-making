#include "mm/rl.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include "mm/strategies.h"

namespace mm {

namespace {
constexpr double kHalf[3] = {0.5, 1.0, 2.0};
constexpr double kSkew[3] = {-0.5, 0.0, 0.5};
constexpr int kDev = 5, kSpread = 2, kVol = 3;
}  // namespace

RlAgent::RlAgent(RlConfig cfg) : cfg_(cfg), rng_(cfg.seed) {
  // Regimes are fixed at 3 by the research pipeline.
  n_states_ = (2 * cfg_.max_lots + 1) * 3 * kDev * kSpread * kVol;
  q_.assign(static_cast<size_t>(n_states_) * kActions, 0.0);
  visited_.assign(q_.size(), 0);
}

void RlAgent::reset() {
  traces_.clear();
  prev_s_ = prev_a_ = -1;
  ep_reward_ = 0;
}

int RlAgent::encode(const Engine& e) const {
  const Signals& s = e.signals();
  int q = static_cast<int>(std::lround(e.state().position / e.config().lot));
  q = std::clamp(q, -cfg_.max_lots, cfg_.max_lots) + cfg_.max_lots;
  int reg = std::clamp(s.regime(), 0, 2);
  double dev = (s.microHmm() - s.mid()) / e.tick();
  int d = dev < -0.25 ? 0 : dev < -0.05 ? 1 : dev <= 0.05 ? 2 : dev <= 0.25 ? 3 : 4;
  int sp = s.spreadTicks() <= 1 ? 0 : 1;
  double sig = s.sigmaTicks();
  int v = sig < cfg_.vol_lo ? 0 : sig < cfg_.vol_hi ? 1 : 2;
  return (((q * 3 + reg) * kDev + d) * kSpread + sp) * kVol + v;
}

int RlAgent::choose(int s) {
  std::uniform_real_distribution<double> u(0, 1);
  if (learning_ && u(rng_) < cfg_.epsilon) return static_cast<int>(rng_() % kActions);
  const double* row = &q_[static_cast<size_t>(s) * kActions];
  return static_cast<int>(std::max_element(row, row + kActions) - row);
}

Quote RlAgent::toQuote(int a, const Engine& e, const Depth& book) const {
  if (a == kActions - 1) return Quote{};  // pull both quotes
  double hs = kHalf[a / 3], sk = kSkew[a % 3];
  // Positive skew shifts both quotes down (lean towards selling).
  return quoteAround(e, book, e.signals().microHmm(), hs + sk, hs - sk);
}

Quote RlAgent::decide(const Engine& e, const Depth& book) {
  if (e.signals().secondsSeen() < cfg_.warmup_s) return Quote{};
  const double unit = e.tick() * e.config().lot;
  const double eq = e.state().equity(), mid = e.signals().mid(), pos = e.state().position;
  int s = encode(e);
  int a = choose(s);

  if (prev_s_ >= 0) {
    double dpnl = (eq - prev_equity_) / unit;
    double spec = prev_pos_ * (mid - prev_mid_) / unit;  // inventory (speculative) PnL
    double lots = pos / e.config().lot;
    double r = dpnl - cfg_.eta * std::max(0.0, spec) - cfg_.kappa * lots * lots;
    ep_reward_ += r;
    if (learning_) {
      size_t sa = static_cast<size_t>(prev_s_) * kActions + prev_a_;
      double target = r + cfg_.gamma * q_[static_cast<size_t>(s) * kActions + a];
      double delta = target - q_[sa];
      // Replacing traces: reset the trace of the visited pair to 1.
      bool found = false;
      for (auto& [idx, z] : traces_)
        if (static_cast<size_t>(idx) == sa) {
          z = 1.0;
          found = true;
        }
      if (!found) traces_.push_back({static_cast<int>(sa), 1.0});
      visited_[sa] = 1;
      const double decay = cfg_.gamma * cfg_.lambda;
      for (auto& [idx, z] : traces_) {
        q_[idx] += cfg_.alpha * delta * z;
        z *= decay;
      }
      traces_.erase(std::remove_if(traces_.begin(), traces_.end(), [](const auto& t) { return t.second < 1e-3; }),
                    traces_.end());
    }
  }
  prev_s_ = s;
  prev_a_ = a;
  prev_equity_ = eq;
  prev_mid_ = mid;
  prev_pos_ = pos;
  return toQuote(a, e, book);
}

bool RlAgent::save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  int n = n_states_, k = kActions;
  out.write(reinterpret_cast<const char*>(&n), sizeof n);
  out.write(reinterpret_cast<const char*>(&k), sizeof k);
  out.write(reinterpret_cast<const char*>(q_.data()), static_cast<std::streamsize>(q_.size() * sizeof(double)));
  return static_cast<bool>(out);
}

bool RlAgent::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  int n = 0, k = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof n);
  in.read(reinterpret_cast<char*>(&k), sizeof k);
  if (n != n_states_ || k != kActions) return false;
  in.read(reinterpret_cast<char*>(q_.data()), static_cast<std::streamsize>(q_.size() * sizeof(double)));
  return static_cast<bool>(in);
}

double RlAgent::coverage() const {
  size_t c = 0;
  for (auto v : visited_) c += v;
  return static_cast<double>(c) / visited_.size();
}

}  // namespace mm
