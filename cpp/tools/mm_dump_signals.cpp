// mm_dump_signals: per-second features and regime posteriors computed online
// by the C++ engine, for parity checks against the Python research code.
#include <cstdio>

#include "common.h"
#include "mm/data.h"
#include "mm/signals.h"

using namespace mm;

int main(int argc, char** argv) {
  auto a = cli::args(argc, argv);
  std::string dir = cli::get(a, "data", "data/processed"), sym = cli::get(a, "symbol", "DOGEUSDT");
  std::string split = cli::get(a, "split", "train");
  double tick = cli::jsonNumber(dir + "/" + sym + ".meta.json", "tick");
  SignalParams sp = SignalParams::load(cli::get(a, "model", "models/" + sym + ".model.txt"));
  MarketData md = MarketData::load(dir + "/" + sym + "." + split, tick);
  Signals s(sp);
  s.keepHistory(true);
  size_t ti = 0;
  std::FILE* out = std::fopen(cli::get(a, "out", "/dev/stdout").c_str(), "w");
  std::fprintf(out, "sec,imb1,imb_deep,ofi1,ofi_deep,tflow5");
  for (int k = 0; k < sp.K; ++k) std::fprintf(out, ",p%d", k);
  std::fprintf(out, "\n");
  size_t printed = 0;
  for (const auto& d : md.depth) {
    while (ti < md.trades.size() && md.trades[ti].ts <= d.ts) s.onTrade(md.trades[ti++]);
    s.onDepth(d);
    // Emit each finalised second once (posterior as of that second).
    while (printed < s.history().size()) {
      const auto& f = s.history()[printed++];
      std::fprintf(out, "%ld,%.10g,%.10g,%.10g,%.10g,%.10g", f.sec, f.imb1, f.imb_deep, f.ofi1, f.ofi_deep, f.tflow5);
      if (printed == s.history().size())
        for (double p : s.regimePosterior()) std::fprintf(out, ",%.8g", p);
      else
        for (int k = 0; k < sp.K; ++k) std::fprintf(out, ",nan");
      std::fprintf(out, "\n");
    }
  }
  std::fclose(out);
  return 0;
}
