#include "mm/data.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mm {

double Depth::qtyAt(int side, double price, double tick) const {
  const auto& px = side > 0 ? bp : ap;
  const auto& q = side > 0 ? bq : aq;
  for (int i = 0; i < kLevels; ++i) {
    if (std::fabs(px[i] - price) < 0.5 * tick) return q[i];
  }
  // Inside the visible range but no level => empty level.
  double worst = px[kLevels - 1];
  bool inside = side > 0 ? price > worst : price < worst;
  return inside ? 0.0 : -1.0;
}

namespace {
std::vector<double> readDoubles(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  auto bytes = static_cast<size_t>(in.tellg());
  std::vector<double> v(bytes / sizeof(double));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
  return v;
}
}  // namespace

MarketData MarketData::load(const std::string& prefix, double tick) {
  MarketData md;
  md.tick = tick;
  auto d = readDoubles(prefix + ".depth.bin");
  constexpr size_t W = 1 + 4 * kLevels;
  md.depth.resize(d.size() / W);
  for (size_t r = 0; r < md.depth.size(); ++r) {
    const double* p = &d[r * W];
    Depth& s = md.depth[r];
    s.ts = p[0];
    for (int i = 0; i < kLevels; ++i) {
      s.bp[i] = p[1 + i];
      s.bq[i] = p[1 + kLevels + i];
      s.ap[i] = p[1 + 2 * kLevels + i];
      s.aq[i] = p[1 + 3 * kLevels + i];
    }
  }
  auto t = readDoubles(prefix + ".trades.bin");
  md.trades.resize(t.size() / 4);
  for (size_t r = 0; r < md.trades.size(); ++r)
    md.trades[r] = {t[4 * r], t[4 * r + 1], t[4 * r + 2], static_cast<int>(t[4 * r + 3])};
  return md;
}

MarketData MarketData::slice(double t0, double t1) const {
  MarketData out;
  out.tick = tick;
  auto dl = std::lower_bound(depth.begin(), depth.end(), t0, [](const Depth& d, double t) { return d.ts < t; });
  auto du = std::lower_bound(depth.begin(), depth.end(), t1, [](const Depth& d, double t) { return d.ts < t; });
  out.depth.assign(dl, du);
  auto tl = std::lower_bound(trades.begin(), trades.end(), t0, [](const Trade& x, double t) { return x.ts < t; });
  auto tu = std::lower_bound(trades.begin(), trades.end(), t1, [](const Trade& x, double t) { return x.ts < t; });
  out.trades.assign(tl, tu);
  return out;
}

}  // namespace mm
