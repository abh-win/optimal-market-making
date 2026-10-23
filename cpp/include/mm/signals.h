#pragma once

// Online signals computed from the market data stream, exactly mirroring the
// research code (research/mmr/prepare.py::second_features):
//   * per-second features: L1 imbalance, deep (levels 2-5) imbalance, L1 and
//     multi-level OFI, 5-second signed trade flow;
//   * causal HMM filter over those features -> regime posterior;
//   * micro-price: Stoikov G*(imbalance, spread) and the regime-augmented
//     G*(imbalance, spread, regime);
//   * EWMA volatility of 1-second mid changes.
// Parameters come from models/<SYMBOL>.model.txt written by research/fit_models.py.

#include <array>
#include <deque>
#include <string>
#include <vector>

#include "mm/data.h"

namespace mm {

struct SignalParams {
  double tick = 0;
  int K = 0, D = 0;                    // HMM states, feature dims
  std::vector<double> feat_scale;      // divide raw features by these
  std::vector<double> pi, A, means, vars;  // A: KxK row-major, means/vars: KxD
  int n_imb = 0, n_spread = 0, n_reg = 0;
  std::vector<double> gstar_plain;     // [spread][imb]
  std::vector<double> gstar_hmm;       // [reg][spread][imb]
  std::vector<double> regime_drift;    // E[dM over next second | regime], ticks
  double fill_A = 0, fill_k = 0;       // lambda(delta) = A exp(-k delta), delta in ticks, per second
  double sigma_ticks = 0;              // std of 1-s mid change, ticks

  static SignalParams load(const std::string& path);
};

struct SecondFeatures {
  long sec = 0;
  double imb1 = 0.5, imb_deep = 0, ofi1 = 0, ofi_deep = 0, tflow5 = 0;
};

class Signals {
 public:
  explicit Signals(const SignalParams& p);

  void onTrade(const Trade& t);
  void onDepth(const Depth& d);

  // Current values (as of the latest snapshot).
  double mid() const { return mid_; }
  int spreadTicks() const { return spread_ticks_; }
  double imb1() const { return imb1_now_; }
  const std::vector<double>& regimePosterior() const { return post_; }
  int regime() const;
  double microPlain() const;  // Stoikov micro-price (price units)
  double microHmm() const;    // regime-augmented micro-price
  double expectedDrift() const;  // posterior-weighted 1-s drift (ticks)
  double sigmaTicks() const { return sigma_ticks_; }  // EWMA, per sqrt(second)
  long secondsSeen() const { return n_seconds_; }
  const std::vector<SecondFeatures>& history() const { return hist_; }  // for parity tests
  void keepHistory(bool on) { keep_hist_ = on; }

 private:
  void finalizeSecond(long sec);
  void hmmStep(const SecondFeatures& f);
  int imbBucket(double imb) const;

  const SignalParams& p_;
  // current snapshot
  double mid_ = 0;
  int spread_ticks_ = 1;
  double imb1_now_ = 0.5;
  // previous snapshot (for OFI)
  bool have_prev_ = false;
  Depth prev_{};
  // accumulation for the current second
  long cur_sec_ = -1;
  double ofi1_acc_ = 0, ofid_acc_ = 0, depth_acc_ = 0;
  int n_snap_ = 0;
  Depth last_in_sec_{};
  SecondFeatures last_feat_;
  std::deque<std::pair<long, std::array<double, 2>>> tflow_;  // (sec, {signed, vol})
  std::vector<double> post_;
  bool post_init_ = false;
  double sigma_ticks_ = 0, last_sec_mid_ = 0;
  long n_seconds_ = 0;
  bool keep_hist_ = false;
  std::vector<SecondFeatures> hist_;
};

}  // namespace mm
