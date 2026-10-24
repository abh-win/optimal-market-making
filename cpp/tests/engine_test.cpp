#include <gtest/gtest.h>

#include "helpers.h"

using namespace mm;
using namespace mmtest;

namespace {
EngineConfig cfg(QueueModel q = QueueModel::RiskAverse, double latency = 0) {
  EngineConfig c;
  c.latency_ms = latency;
  c.decision_ms = 1e12;  // decide once at the first snapshot
  c.lot = 3;
  c.max_position = 10;
  c.queue = q;
  return c;
}
}  // namespace

TEST(Engine, FifoQueueFillsOnlyAfterQueueAhead) {
  MarketData md;
  md.tick = 1;
  md.depth = {book(0, 100, 10, 101, 10), book(1000, 100, 4, 101, 10), book(2000, 100, 0, 101, 10)};
  md.trades = {{500, 100, 6, -1}, {1500, 100, 6, -1}};  // two sells of 6 at our bid
  auto sp = trivialParams();
  Engine e(md, sp, cfg());
  FixedQuotes s(true, 100, false, 0);
  e.run(s);
  // queue ahead 10 -> after first trade 4 -> second trade: 6 - 4 = 2 fill (lot 3 -> partial)
  ASSERT_EQ(e.fills().size(), 1u);
  EXPECT_DOUBLE_EQ(e.fills()[0].qty, 2);
  EXPECT_DOUBLE_EQ(e.state().position, 2);
  EXPECT_DOUBLE_EQ(e.state().cash, -200);
}

TEST(Engine, CancellationModels) {
  // Depth at our level drops 10 -> 5 without trades: 5 cancellations.
  MarketData md;
  md.tick = 1;
  md.depth = {book(0, 100, 10, 101, 10), book(1000, 100, 5, 101, 10)};
  md.trades = {{1500, 100, 3, -1}};  // a sell of 3 after the cancels
  auto sp = trivialParams();
  auto run = [&](QueueModel q) {
    Engine e(md, sp, cfg(q));
    FixedQuotes s(true, 100, false, 0);
    e.run(s);
    return e.state().position;
  };
  // risk-averse: cancels behind us -> ahead = min(10, visible 5) = 5 -> trade of 3 fills nothing
  EXPECT_DOUBLE_EQ(run(QueueModel::RiskAverse), 0);
  // proportional: x = 1 (we are at the back) -> all 5 cancels ahead -> ahead 5 -> no fill either
  EXPECT_DOUBLE_EQ(run(QueueModel::Proportional), 0);
}

TEST(Engine, ProportionalCancelsAdvanceUsInTheMiddleOfTheQueue) {
  MarketData md;
  md.tick = 1;
  // We join behind 10; then 10 more arrive behind us (20), then 10 cancel (-> 10).
  md.depth = {book(0, 100, 10, 101, 10), book(1000, 100, 20, 101, 10), book(2000, 100, 10, 101, 10)};
  md.trades = {{2500, 100, 6, -1}};
  auto sp = trivialParams();
  auto pos = [&](QueueModel q) {
    Engine e(md, sp, cfg(q));
    FixedQuotes s(true, 100, false, 0);
    e.run(s);
    return e.state().position;
  };
  EXPECT_DOUBLE_EQ(pos(QueueModel::RiskAverse), 0);    // still 10 ahead
  EXPECT_DOUBLE_EQ(pos(QueueModel::Proportional), 1);  // x = 0.5 -> 5 cancels ahead -> 5 ahead -> 6-5 = 1
  // power2 with x = 0.5 is also 0.5
  EXPECT_DOUBLE_EQ(pos(QueueModel::PowerProb2), 1);
}

TEST(Engine, TradeThroughAndCrossingFillCompletely) {
  MarketData md;
  md.tick = 1;
  md.depth = {book(0, 100, 10, 101, 10), book(1000, 98, 5, 99, 5)};
  md.trades = {{500, 99, 1, -1}};  // sell at 99 < our bid 100: traded through
  auto sp = trivialParams();
  Engine e(md, sp, cfg());
  FixedQuotes s(true, 100, true, 103);
  e.run(s);
  ASSERT_EQ(e.fills().size(), 1u);
  EXPECT_DOUBLE_EQ(e.fills()[0].qty, 3);

  MarketData md2;
  md2.tick = 1;
  md2.depth = {book(0, 100, 10, 101, 10), book(1000, 103, 5, 104, 5)};  // book jumps through our ask
  Engine e2(md2, sp, cfg());
  FixedQuotes s2(false, 0, true, 102);
  e2.run(s2);
  ASSERT_EQ(e2.fills().size(), 1u);
  EXPECT_EQ(e2.fills()[0].side, -1);
  EXPECT_DOUBLE_EQ(e2.fills()[0].price, 102);
}

TEST(Engine, LatencyDelaysOrderEntry) {
  MarketData md;
  md.tick = 1;
  md.depth = {book(0, 100, 0, 101, 10), book(100, 100, 0, 101, 10)};
  md.trades = {{30, 100, 5, -1}, {80, 100, 5, -1}};
  auto sp = trivialParams();
  Engine e(md, sp, cfg(QueueModel::RiskAverse, 50));
  FixedQuotes s(true, 100, false, 0);
  e.run(s);
  // Order live at t=50: the trade at 30 misses us, the one at 80 fills us (empty queue).
  ASSERT_EQ(e.fills().size(), 1u);
  EXPECT_DOUBLE_EQ(e.fills()[0].ts, 80);
}

class Requoter : public Strategy {
 public:
  std::string name() const override { return "requote"; }
  Quote decide(const Engine&, const Depth& d) override {
    Quote q;
    q.bid_on = true;
    q.bid = d.ts < 1 ? 100 : 99;  // move the bid down at the second decision
    return q;
  }
};

TEST(Engine, StaleQuoteCanFillWhileCancelIsInFlight) {
  MarketData md;
  md.tick = 1;
  md.depth = {book(0, 100, 0, 101, 10), book(1000, 100, 0, 101, 10), book(2000, 100, 0, 101, 10)};
  md.trades = {{1010, 100, 5, -1}};  // arrives 10ms after we decided to requote (latency 50)
  auto sp = trivialParams();
  EngineConfig c = cfg(QueueModel::RiskAverse, 50);
  c.decision_ms = 1000;
  Engine e(md, sp, c);
  Requoter s;
  e.run(s);
  ASSERT_EQ(e.fills().size(), 1u);
  EXPECT_DOUBLE_EQ(e.fills()[0].price, 100);  // the old (stale) order was hit
}

TEST(Engine, InventoryLimitSuppressesQuotes) {
  MarketData md;
  md.tick = 1;
  md.depth = {book(0, 100, 0, 101, 0), book(1000, 100, 0, 101, 0), book(2000, 100, 0, 101, 0)};
  md.trades = {{500, 100, 100, -1}, {1500, 100, 100, -1}};
  auto sp = trivialParams();
  EngineConfig c = cfg();
  c.max_position = 1;  // one lot
  c.decision_ms = 1000;
  Engine e(md, sp, c);
  FixedQuotes s(true, 100, false, 0);
  e.run(s);
  EXPECT_DOUBLE_EQ(e.state().position, 3);  // filled once, then the bid is pulled at the limit
}

TEST(Engine, FeesAndEquity) {
  MarketData md;
  md.tick = 1;
  md.depth = {book(0, 100, 0, 101, 0), book(1000, 102, 0, 103, 0)};
  md.trades = {{500, 100, 3, -1}};
  auto sp = trivialParams();
  EngineConfig c = cfg();
  c.maker_fee = 0.001;
  Engine e(md, sp, c);
  FixedQuotes s(true, 100, false, 0);
  e.run(s);
  EXPECT_NEAR(e.state().fees, 0.3, 1e-12);                       // 3 * 100 * 0.1%
  EXPECT_NEAR(e.state().equity(), -300 - 0.3 + 3 * 102.5, 1e-9);  // marked at mid 102.5
}
