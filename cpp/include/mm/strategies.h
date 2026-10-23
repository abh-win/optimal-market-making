#pragma once

// Quoting strategies. All prices are computed in ticks relative to a fair
// value ("center") and rounded outward (bid down, ask up).
//
// FixedSpread     join the best bid/ask, step back one tick on the side that
//                 would increase inventory.
// AvellanedaStoikov (2008)  reservation price r = c - q*gamma*sigma^2*tau,
//                 total spread gamma*sigma^2*tau + (2/gamma) ln(1 + gamma/k).
// GLFT            Gueant-Lehalle-Fernandez-Tapia (2013) closed-form asymptotic
//                 quotes with inventory limits:
//                   delta_b(q) = c1 + (2q+1)/2 * sigma * c2,  delta_a(q) = c1 - (2q-1)/2 * sigma * c2,
//                   c1 = ln(1 + gamma/k) / gamma,  c2 = sqrt(gamma/(2Ak) (1 + gamma/k)^(1 + k/gamma)).
//                 Center = mid, Stoikov micro-price, or the HMM micro-price.
// HmmGLFT         GLFT around the HMM micro-price plus (a) an alpha skew by the
//                 regime-expected drift (Cartea-Jaimungal style signal) and
//                 (b) an adverse-selection guard: in a "buy pressure" regime the
//                 ask is widened (informed buyers lift asks), and vice versa.

#include <string>

#include "mm/engine.h"

namespace mm {

enum class Center { Mid, MicroPlain, MicroHmm };

struct GlftParams {
  double gamma = 0.1;   // risk aversion (per tick of inventory risk)
  double A = 0, k = 0;  // fill intensity lambda(delta) = A exp(-k delta), delta in ticks, per second
  double sigma = 0;     // ticks per sqrt(second); <= 0 => use online EWMA estimate
  Center center = Center::Mid;
};

class FixedSpread final : public Strategy {
 public:
  std::string name() const override { return "fixed-spread (join best)"; }
  Quote decide(const Engine& e, const Depth& book) override;
};

class AvellanedaStoikov final : public Strategy {
 public:
  AvellanedaStoikov(double gamma, double k, double tau_s) : gamma_(gamma), k_(k), tau_(tau_s) {}
  std::string name() const override { return "Avellaneda-Stoikov"; }
  Quote decide(const Engine& e, const Depth& book) override;

 private:
  double gamma_, k_, tau_;
};

class Glft final : public Strategy {
 public:
  explicit Glft(GlftParams p) : p_(p) {}
  std::string name() const override;
  Quote decide(const Engine& e, const Depth& book) override;

 private:
  GlftParams p_;
};

struct HmmGuardParams {
  double alpha_horizon_s = 5;   // center += drift_per_second * horizon
  double guard_threshold = 0.6; // posterior of an adverse regime that triggers the guard
  double guard_ticks = 1;       // extra distance on the threatened side
};

class HmmGlft final : public Strategy {
 public:
  HmmGlft(GlftParams p, HmmGuardParams g) : p_(p), g_(g) { p_.center = Center::MicroHmm; }
  std::string name() const override { return "HMM-GLFT (micro-price + alpha + AS guard)"; }
  Quote decide(const Engine& e, const Depth& book) override;

 private:
  GlftParams p_;
  HmmGuardParams g_;
};

// Shared helpers.
double glftHalfSpread(double gamma, double k);                       // c1
double glftSkew(double gamma, double k, double A, double sigma);     // sigma * c2
double centerPrice(const Engine& e, Center c);
Quote quoteAround(const Engine& e, const Depth& book, double center, double bid_depth_ticks, double ask_depth_ticks);

}  // namespace mm
