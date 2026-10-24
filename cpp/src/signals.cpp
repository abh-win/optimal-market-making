#include "mm/signals.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mm {

namespace {
std::vector<double> readVec(std::istream& in, size_t n) {
  std::vector<double> v(n);
  for (auto& x : v) in >> x;
  return v;
}
}  // namespace

SignalParams SignalParams::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open model file " + path);
  SignalParams p;
  std::string key;
  while (in >> key) {
    if (key == "tick") in >> p.tick;
    else if (key == "hmm") in >> p.K >> p.D;
    else if (key == "feat_scale") p.feat_scale = readVec(in, p.D);
    else if (key == "pi") p.pi = readVec(in, p.K);
    else if (key == "A") p.A = readVec(in, p.K * p.K);
    else if (key == "means") p.means = readVec(in, p.K * p.D);
    else if (key == "vars") p.vars = readVec(in, p.K * p.D);
    else if (key == "micro") in >> p.n_imb >> p.n_spread >> p.n_reg;
    else if (key == "gstar_plain") p.gstar_plain = readVec(in, p.n_spread * p.n_imb);
    else if (key == "gstar_hmm") p.gstar_hmm = readVec(in, p.n_reg * p.n_spread * p.n_imb);
    else if (key == "drift") p.regime_drift = readVec(in, p.K);
    else if (key == "fill") in >> p.fill_A >> p.fill_k;
    else if (key == "sigma") in >> p.sigma_ticks;
    else {
      std::string rest;
      std::getline(in, rest);  // ignore unknown keys
    }
  }
  if (p.tick <= 0 || p.K <= 0) throw std::runtime_error("incomplete model file " + path);
  return p;
}

Signals::Signals(const SignalParams& p) : p_(p), post_(p.pi) { sigma_ticks_ = p.sigma_ticks; }

int Signals::regime() const { return static_cast<int>(std::max_element(post_.begin(), post_.end()) - post_.begin()); }

void Signals::onTrade(const Trade& t) {
  long sec = static_cast<long>(std::floor(t.ts / 1000.0));
  if (!tflow_.empty() && tflow_.back().first == sec) {
    tflow_.back().second[0] += t.side * t.qty;
    tflow_.back().second[1] += t.qty;
  } else {
    tflow_.push_back({sec, {t.side * t.qty, t.qty}});
  }
}

void Signals::onDepth(const Depth& d) {
  long sec = static_cast<long>(std::floor(d.ts / 1000.0));
  if (cur_sec_ >= 0 && sec != cur_sec_) {
    finalizeSecond(cur_sec_);
    // Seconds without snapshots: features forward-filled, OFI zero (as in pandas ffill/fillna).
    for (long s = cur_sec_ + 1; s < sec; ++s) {
      ofi1_acc_ = ofid_acc_ = 0;
      n_snap_ = 0;
      finalizeSecond(s);
    }
    ofi1_acc_ = ofid_acc_ = depth_acc_ = 0;
    n_snap_ = 0;
  }
  cur_sec_ = sec;

  if (have_prev_) {
    auto lvl = [&](int k) {
      double b0 = prev_.bp[k], b1 = d.bp[k], q0b = prev_.bq[k], q1b = d.bq[k];
      double a0 = prev_.ap[k], a1 = d.ap[k], q0a = prev_.aq[k], q1a = d.aq[k];
      return (b1 >= b0) * q1b - (b1 <= b0) * q0b - (a1 <= a0) * q1a + (a1 >= a0) * q0a;
    };
    ofi1_acc_ += lvl(0);
    for (int k = 1; k < 5; ++k) ofid_acc_ += lvl(k);
  }
  double depth = 0;
  for (int k = 0; k < 5; ++k) depth += d.bq[k] + d.aq[k];
  depth_acc_ += depth;
  ++n_snap_;
  last_in_sec_ = d;
  prev_ = d;
  have_prev_ = true;

  mid_ = d.mid();
  spread_ticks_ = static_cast<int>(std::lround((d.ask() - d.bid()) / p_.tick));
  imb1_now_ = d.bq[0] / (d.bq[0] + d.aq[0]);
}

void Signals::finalizeSecond(long sec) {
  SecondFeatures f = last_feat_;
  f.sec = sec;
  if (n_snap_ > 0) {
    const Depth& d = last_in_sec_;
    f.imb1 = d.bq[0] / (d.bq[0] + d.aq[0]);
    double bd = 0, ad = 0;
    for (int k = 1; k < 5; ++k) {
      bd += d.bq[k];
      ad += d.aq[k];
    }
    f.imb_deep = (bd - ad) / (bd + ad);
    double mean_depth = depth_acc_ / n_snap_;
    f.ofi1 = ofi1_acc_ / mean_depth * 10;
    f.ofi_deep = ofid_acc_ / mean_depth * 10;
  } else {
    f.ofi1 = f.ofi_deep = 0;
  }
  // 5-second trade flow over seconds [sec-4, sec].
  while (!tflow_.empty() && tflow_.front().first < sec - 4) tflow_.pop_front();
  double s = 0, v = 0;
  for (const auto& e : tflow_) {
    if (e.first <= sec) {
      s += e.second[0];
      v += e.second[1];
    }
  }
  f.tflow5 = v > 0 ? s / v : 0.0;
  last_feat_ = f;
  if (keep_hist_) hist_.push_back(f);
  hmmStep(f);

  // EWMA of squared 1-second mid changes (half-life ~ 5 minutes).
  double m = n_snap_ > 0 ? last_in_sec_.mid() : last_sec_mid_;
  if (n_seconds_ > 0) {
    double dm = (m - last_sec_mid_) / p_.tick;
    double var = sigma_ticks_ * sigma_ticks_;
    var = 0.9977 * var + 0.0023 * dm * dm;
    sigma_ticks_ = std::sqrt(var);
  }
  last_sec_mid_ = m;
  ++n_seconds_;
}

void Signals::hmmStep(const SecondFeatures& f) {
  const int K = p_.K, D = p_.D;
  double x[5] = {2 * f.imb1 - 1, f.imb_deep, f.ofi1, f.ofi_deep, f.tflow5};
  std::vector<double> logb(K);
  for (int k = 0; k < K; ++k) {
    double s = 0;
    for (int j = 0; j < D; ++j) {
      double z = x[j] / p_.feat_scale[j] - p_.means[k * D + j];
      double var = p_.vars[k * D + j];
      s += z * z / var + std::log(var);
    }
    logb[k] = -0.5 * s;
  }
  double mx = *std::max_element(logb.begin(), logb.end());
  std::vector<double> a(K, 0.0);
  double tot = 0;
  for (int k = 0; k < K; ++k) {
    double prior = 0;
    if (!post_init_) prior = p_.pi[k];
    else
      for (int i = 0; i < K; ++i) prior += post_[i] * p_.A[i * K + k];
    a[k] = prior * std::exp(logb[k] - mx);
    tot += a[k];
  }
  for (int k = 0; k < K; ++k) post_[k] = a[k] / tot;
  post_init_ = true;
}

int Signals::imbBucket(double imb) const {
  int b = static_cast<int>(imb * p_.n_imb);
  return std::clamp(b, 0, p_.n_imb - 1);
}

double Signals::microPlain() const {
  int s = std::clamp(spread_ticks_, 1, p_.n_spread) - 1;
  return mid_ + p_.gstar_plain[s * p_.n_imb + imbBucket(imb1_now_)] * p_.tick;
}

double Signals::microHmm() const {
  int s = std::clamp(spread_ticks_, 1, p_.n_spread) - 1;
  // Posterior-weighted adjustment across regimes.
  double g = 0;
  for (int r = 0; r < p_.n_reg; ++r)
    g += post_[r] * p_.gstar_hmm[(r * p_.n_spread + s) * p_.n_imb + imbBucket(imb1_now_)];
  return mid_ + g * p_.tick;
}

double Signals::expectedDrift() const {
  double g = 0;
  for (int k = 0; k < p_.K; ++k) g += post_[k] * p_.regime_drift[k];
  return g;
}

}  // namespace mm
