#include <gtest/gtest.h>

#include <cmath>

#include "helpers.h"
#include "mm/strategies.h"

using namespace mm;
using namespace mmtest;

TEST(Glft, ClosedFormProperties) {
  double c1 = glftHalfSpread(0.1, 1.5);
  EXPECT_NEAR(c1, std::log(1 + 0.1 / 1.5) / 0.1, 1e-12);
  // gamma -> 0: half spread -> 1/k (the risk-neutral monopolist spread).
  EXPECT_NEAR(glftHalfSpread(1e-8, 2.0), 0.5, 1e-6);
  // Skew grows with risk aversion and volatility.
  EXPECT_GT(glftSkew(0.3, 1.5, 1.0, 1.0), glftSkew(0.1, 1.5, 1.0, 1.0));
  EXPECT_NEAR(glftSkew(0.1, 1.5, 1.0, 2.0), 2 * glftSkew(0.1, 1.5, 1.0, 1.0), 1e-12);
}

TEST(Glft, QuotesSkewAgainstInventory) {
  // A long position must lower both quotes (more eager to sell, less to buy).
  MarketData md;
  md.tick = 1;
  md.depth = {book(0, 1000, 10, 1001, 10)};
  auto sp = trivialParams();
  EngineConfig c;
  c.lot = 1;
  c.max_position = 10;
  GlftParams p;
  p.gamma = 0.5;
  p.A = 1;
  p.k = 0.5;
  p.sigma = 3;
  Glft g(p);
  Engine e(md, sp, c);
  FixedQuotes dummy(false, 0, false, 0);
  e.run(dummy);  // load one snapshot into the signals
  Quote flat = g.decide(e, md.depth[0]);
  EXPECT_LT(flat.bid, flat.ask);
  // Symmetric around the mid when flat.
  EXPECT_NEAR(1000.5 - flat.bid, flat.ask - 1000.5, 1.0 + 1e-9);
}
