#include "mm/strategies.h"

#include <algorithm>
#include <cmath>

namespace mm {

double glftHalfSpread(double gamma, double k) { return std::log1p(gamma / k) / gamma; }

double glftSkew(double gamma, double k, double A, double sigma) {
  return sigma * std::sqrt(gamma / (2 * A * k) * std::pow(1 + gamma / k, 1 + k / gamma));
}

double centerPrice(const Engine& e, Center c) {
  switch (c) {
    case Center::Mid: return e.signals().mid();
    case Center::MicroPlain: return e.signals().microPlain();
    case Center::MicroHmm: return e.signals().microHmm();
  }
  return e.signals().mid();
}

Quote quoteAround(const Engine& e, const Depth& book, double center, double bid_depth, double ask_depth) {
  Quote q;
  q.bid_on = q.ask_on = true;
  const double tk = e.tick();
  // Bid rounded down, ask rounded up to the tick grid; never better than one tick inside the touch.
  q.bid = std::floor((center - bid_depth * tk) / tk + 1e-9) * tk;
  q.ask = std::ceil((center + ask_depth * tk) / tk - 1e-9) * tk;
  q.bid = std::min(q.bid, book.ask() - tk);
  q.ask = std::max(q.ask, book.bid() + tk);
  return q;
}

Quote FixedSpread::decide(const Engine& e, const Depth& book) {
  Quote q;
  q.bid_on = q.ask_on = true;
  const double tk = e.tick();
  double lots = e.state().position / e.config().lot;
  q.bid = book.bid() - (lots > 0 ? tk : 0);
  q.ask = book.ask() + (lots < 0 ? tk : 0);
  return q;
}

Quote AvellanedaStoikov::decide(const Engine& e, const Depth& book) {
  double sigma = e.signals().sigmaTicks();
  double q = e.state().position / e.config().lot;
  double var_tau = sigma * sigma * tau_;
  double r = -q * gamma_ * var_tau;  // reservation offset in ticks
  double spread = gamma_ * var_tau + (2 / gamma_) * std::log1p(gamma_ / k_);
  return quoteAround(e, book, e.signals().mid(), spread / 2 - r, spread / 2 + r);
}

std::string Glft::name() const {
  switch (p_.center) {
    case Center::Mid: return "GLFT (mid)";
    case Center::MicroPlain: return "GLFT (Stoikov micro-price)";
    case Center::MicroHmm: return "GLFT (HMM micro-price)";
  }
  return "GLFT";
}

Quote Glft::decide(const Engine& e, const Depth& book) {
  double sigma = p_.sigma > 0 ? p_.sigma : e.signals().sigmaTicks();
  double q = e.state().position / e.config().lot;
  double c1 = glftHalfSpread(p_.gamma, p_.k);
  double sk = glftSkew(p_.gamma, p_.k, p_.A, sigma);
  double bid_depth = c1 + (2 * q + 1) / 2 * sk;
  double ask_depth = c1 - (2 * q - 1) / 2 * sk;
  return quoteAround(e, book, centerPrice(e, p_.center), bid_depth, ask_depth);
}

Quote HmmGlft::decide(const Engine& e, const Depth& book) {
  const Signals& s = e.signals();
  double sigma = p_.sigma > 0 ? p_.sigma : s.sigmaTicks();
  double q = e.state().position / e.config().lot;
  double c1 = glftHalfSpread(p_.gamma, p_.k);
  double sk = glftSkew(p_.gamma, p_.k, p_.A, sigma);
  const double tk = e.tick();
  double center = s.microHmm() + s.expectedDrift() * g_.alpha_horizon_s * tk;
  double bid_depth = c1 + (2 * q + 1) / 2 * sk;
  double ask_depth = c1 - (2 * q - 1) / 2 * sk;
  const auto& post = s.regimePosterior();
  // Regimes are sorted by mean signed flow: 0 = sell pressure, K-1 = buy pressure.
  if (post.front() > g_.guard_threshold) bid_depth += g_.guard_ticks;  // sellers informed: bids get picked off
  if (post.back() > g_.guard_threshold) ask_depth += g_.guard_ticks;   // buyers informed: asks get picked off
  return quoteAround(e, book, center, bid_depth, ask_depth);
}

}  // namespace mm
