#pragma once
#include "mm/data.h"
#include "mm/engine.h"
#include "mm/signals.h"

namespace mmtest {
using namespace mm;

inline SignalParams trivialParams(double tick = 1.0) {
  SignalParams p;
  p.tick = tick;
  p.K = 1;
  p.D = 5;
  p.feat_scale.assign(5, 1.0);
  p.pi = {1.0};
  p.A = {1.0};
  p.means.assign(5, 0.0);
  p.vars.assign(5, 1.0);
  p.n_imb = 1;
  p.n_spread = 1;
  p.n_reg = 1;
  p.gstar_plain = {0.0};
  p.gstar_hmm = {0.0};
  p.regime_drift = {0.0};
  p.fill_A = 1;
  p.fill_k = 1;
  p.sigma_ticks = 1;
  return p;
}

// Book with best bid 100 (qty bq1) / ask 101 (qty aq1), deeper levels 10 each.
inline Depth book(double ts, double bid, double bq1, double ask, double aq1) {
  Depth d;
  d.ts = ts;
  for (int i = 0; i < kLevels; ++i) {
    d.bp[i] = bid - i;
    d.ap[i] = ask + i;
    d.bq[i] = i == 0 ? bq1 : 10;
    d.aq[i] = i == 0 ? aq1 : 10;
  }
  return d;
}

// Strategy that always quotes fixed prices (or nothing).
class FixedQuotes : public Strategy {
 public:
  FixedQuotes(bool b, double bp, bool a, double ap) { q_.bid_on = b; q_.bid = bp; q_.ask_on = a; q_.ask = ap; }
  std::string name() const override { return "fixed"; }
  Quote decide(const Engine&, const Depth&) override { return q_; }
  Quote q_;
};
}  // namespace mmtest
