#pragma once

// Reinforcement-learning market maker (after Spooner, Fearnley, Savani &
// Koukorinis 2018, "Market Making via Reinforcement Learning").
//
// Algorithm   on-policy SARSA(lambda), tabular, replacing eligibility traces,
//             epsilon-greedy exploration with decay.
// State       inventory (lots, -Q..Q) x HMM regime (argmax filtered posterior)
//             x micro-price deviation from mid (5 buckets) x spread (1 / >=2 ticks)
//             x volatility bucket (3, thresholds from training data).
// Actions     quote around the HMM micro-price with half-spread in
//             {0.5, 1, 2} ticks and skew in {-0.5, 0, +0.5} ticks, or pull quotes.
// Reward      asymmetrically dampened PnL: dPnL - eta * max(0, q * dMid)
//             (no credit for speculative inventory gains), minus kappa * q^2;
//             in units of tick * lot.
// Inventory   hard limit |q| <= Q enforced by the engine (quotes that would
//             breach are never sent), so the agent learns within the limit.

#include <random>
#include <string>
#include <vector>

#include "mm/engine.h"

namespace mm {

struct RlConfig {
  int max_lots = 5;
  double alpha = 0.05, gamma = 0.99, lambda = 0.8;
  double epsilon = 0.1;
  double eta = 0.5;      // asymmetric dampening
  double kappa = 0.02;   // inventory penalty per lot^2 per decision
  double vol_lo = 0.5, vol_hi = 1.5;  // sigma bucket edges (ticks / sqrt s)
  double warmup_s = 120; // seconds of data before the agent starts acting
  uint64_t seed = 1;
};

class RlAgent final : public Strategy {
 public:
  explicit RlAgent(RlConfig cfg);
  std::string name() const override { return "RL (SARSA-lambda) on HMM micro-price"; }
  void reset() override;  // new episode: clears traces and the previous transition
  Quote decide(const Engine& e, const Depth& book) override;

  void setLearning(bool on) { learning_ = on; }
  void setEpsilon(double eps) { cfg_.epsilon = eps; }
  double episodeReward() const { return ep_reward_; }
  int numStates() const { return n_states_; }
  static constexpr int kActions = 10;

  bool save(const std::string& path) const;
  bool load(const std::string& path);
  // Fraction of (state, action) pairs ever updated.
  double coverage() const;

 private:
  int encode(const Engine& e) const;
  int choose(int s);
  Quote toQuote(int a, const Engine& e, const Depth& book) const;

  RlConfig cfg_;
  int n_states_;
  std::vector<double> q_;
  std::vector<unsigned char> visited_;
  std::vector<std::pair<int, double>> traces_;
  std::mt19937_64 rng_;
  bool learning_ = true;
  int prev_s_ = -1, prev_a_ = -1;
  double prev_equity_ = 0, prev_mid_ = 0, prev_pos_ = 0;
  double ep_reward_ = 0;
};

}  // namespace mm
