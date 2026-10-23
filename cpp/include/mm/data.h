#pragma once

// Market data for the backtester: 10-level depth snapshots (Binance depth20@100ms,
// top 10 kept) and individual trades, both stamped with exchange transaction
// time in milliseconds. Produced by research/mmr/prepare.py.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mm {

constexpr int kLevels = 10;

struct Depth {
  double ts;  // ms
  std::array<double, kLevels> bp, bq, ap, aq;
  double bid() const { return bp[0]; }
  double ask() const { return ap[0]; }
  double mid() const { return 0.5 * (bp[0] + ap[0]); }
  // Visible quantity resting at `price` on the bid (side=+1) or ask (side=-1), or -1 if
  // the price is outside the 10 visible levels.
  double qtyAt(int side, double price, double tick) const;
};

struct Trade {
  double ts;
  double price;
  double qty;
  int side;  // +1 buyer-initiated (lifts asks), -1 seller-initiated (hits bids)
};

struct MarketData {
  std::vector<Depth> depth;
  std::vector<Trade> trades;
  double tick = 0;

  // Loads <prefix>.depth.bin and <prefix>.trades.bin.
  static MarketData load(const std::string& prefix, double tick);
  // Sub-range [t0, t1) in ms (copies).
  MarketData slice(double t0, double t1) const;
};

}  // namespace mm
