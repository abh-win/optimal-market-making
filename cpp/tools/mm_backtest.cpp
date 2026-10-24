// mm_backtest: evaluate market-making strategies on recorded data.
//
//   mm_backtest --data data/processed --symbol DOGEUSDT [--model models/DOGEUSDT.model.txt]
//               [--lot-usd 100] [--max-pos 5] [--latency 20] [--queue power2] [--fee 0]
//               [--rl models/DOGEUSDT.rl.bin] [--csv results/DOGEUSDT.csv] [--sensitivity]
//
// Strategy parameters (gamma) are tuned on the TRAIN split by PnL; every number
// reported is then measured on the held-out TEST split.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>

#include "common.h"
#include "mm/engine.h"
#include "mm/rl.h"
#include "mm/strategies.h"

using namespace mm;

namespace {

struct Run {
  std::string name;
  Metrics m;
  std::vector<double> regime_markout10;  // mean 10s markout (ticks) by regime at fill
  std::vector<long> regime_fills;
};

Run runOne(const MarketData& md, const SignalParams& sp, const EngineConfig& cfg, Strategy& s) {
  Engine e(md, sp, cfg);
  e.run(s);
  Run r{s.name(), computeMetrics(e, md.tick), std::vector<double>(sp.K, 0), std::vector<long>(sp.K, 0)};
  for (const auto& f : e.fills()) {
    if (f.markout.size() < 2) continue;
    r.regime_markout10[f.regime] += f.markout[1] / md.tick;
    r.regime_fills[f.regime]++;
  }
  for (int k = 0; k < sp.K; ++k) r.regime_markout10[k] /= std::max(1L, r.regime_fills[k]);
  return r;
}

void printHeader() {
  std::printf("%-44s %10s %8s %7s %9s %7s %9s %9s %8s %7s %7s %8s %8s %8s\n", "strategy", "PnL", "PnL/h", "Sharpe/d",
              "MaxDD", "fills", "spread$", "invent$", "fees$", "|pos|", "@limit", "mo1s", "mo10s", "mo60s");
}

void printRow(const Run& r) {
  const Metrics& m = r.m;
  std::printf("%-44s %10.2f %8.2f %7.2f %9.2f %7ld %9.2f %9.2f %8.2f %7.2f %6.0f%% %8.3f %8.3f %8.3f\n",
              r.name.c_str(), m.pnl, m.pnl_per_hour, m.sharpe_hourly, m.max_drawdown, m.fills, m.spread_capture,
              m.inventory_pnl, m.fees, m.mean_abs_position, 100 * m.time_at_limit, m.markout_mean[0], m.markout_mean[1],
              m.markout_mean[2]);
}

}  // namespace

int main(int argc, char** argv) {
  auto a = cli::args(argc, argv);
  std::string dir = cli::get(a, "data", "data/processed"), sym = cli::get(a, "symbol", "DOGEUSDT");
  std::string model = cli::get(a, "model", "models/" + sym + ".model.txt");
  double tick = cli::jsonNumber(dir + "/" + sym + ".meta.json", "tick");
  SignalParams sp = SignalParams::load(model);
  MarketData train = MarketData::load(dir + "/" + sym + ".train", tick);
  MarketData test = MarketData::load(dir + "/" + sym + ".test", tick);

  EngineConfig cfg;
  double lot_usd = std::stod(cli::get(a, "lot-usd", "100"));
  cfg.lot = std::max(1.0, std::round(lot_usd / train.depth.front().mid()));
  cfg.max_position = std::stod(cli::get(a, "max-pos", "5"));
  cfg.latency_ms = std::stod(cli::get(a, "latency", "20"));
  cfg.queue = parseQueueModel(cli::get(a, "queue", "power2"));
  cfg.maker_fee = std::stod(cli::get(a, "fee", "0"));
  cfg.decision_ms = 100;
  cfg.keep_ticks = std::stoi(cli::get(a, "keep-ticks", "0"));

  std::printf("%s: tick %g, lot %g units (~$%.0f), max position %g lots, latency %g ms, queue %s, maker fee %.4f%%, "
              "keep-ticks %d\n",
              sym.c_str(), tick, cfg.lot, lot_usd, cfg.max_position, cfg.latency_ms, toString(cfg.queue),
              cfg.maker_fee * 100, cfg.keep_ticks);
  std::printf("train %.2f h (%zu snapshots, %zu trades) | test %.2f h (%zu snapshots, %zu trades)\n",
              (train.depth.back().ts - train.depth.front().ts) / 3.6e6, train.depth.size(), train.trades.size(),
              (test.depth.back().ts - test.depth.front().ts) / 3.6e6, test.depth.size(), test.trades.size());
  std::printf("fill model A=%.3f/s k=%.3f/tick, sigma=%.3f ticks/sqrt(s)\n\n", sp.fill_A, sp.fill_k, sp.sigma_ticks);

  // ---- tune on TRAIN. Every quoting configuration can lose money on train, in
  // which case maximising PnL degenerates to "do not quote" (huge gamma, zero
  // fills). Strategies are therefore compared at comparable participation:
  // among settings that trade at least `min_part` x the fixed-spread baseline's
  // train volume, pick the highest train PnL. The whole gamma frontier (train
  // and test) is also written to CSV.
  const std::vector<double> gammas = {0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0};
  const double min_part = std::stod(cli::get(a, "min-participation", "0.25"));
  long base_fills;
  {
    FixedSpread fs;
    base_fills = runOne(train, sp, cfg, fs).m.fills;
  }
  const long min_fills = static_cast<long>(min_part * base_fills);
  std::unique_ptr<std::ofstream> frontier;
  if (a.count("frontier")) {
    frontier = std::make_unique<std::ofstream>(a.at("frontier"));
    *frontier << "symbol,family,gamma,guard,train_pnl,train_fills,test_pnl,test_fills,test_markout10,keep_ticks\n";
  }
  auto tune = [&](const std::string& family, const std::function<std::unique_ptr<Strategy>(double)>& make,
                  double guard = -1) {
    double best_g = gammas[0], best = -1e300, most_fills_g = gammas[0];
    long most_fills = -1;
    for (double g : gammas) {
      auto s = make(g);
      Metrics tr = runOne(train, sp, cfg, *s).m;
      if (tr.fills >= min_fills && tr.pnl > best) {
        best = tr.pnl;
        best_g = g;
      }
      if (tr.fills > most_fills) {
        most_fills = tr.fills;
        most_fills_g = g;
      }
      if (frontier) {
        auto s2 = make(g);
        Metrics te = runOne(test, sp, cfg, *s2).m;
        *frontier << sym << ",\"" << family << "\"," << g << "," << guard << "," << tr.pnl << "," << tr.fills << ","
                  << te.pnl << "," << te.fills << "," << te.markout_mean[1] << "," << cfg.keep_ticks << "\n";
      }
    }
    return best > -1e299 ? best_g : most_fills_g;
  };
  auto glft = [&](Center c) {
    return [&, c](double g) {
      GlftParams p;
      p.gamma = g;
      p.A = sp.fill_A;
      p.k = sp.fill_k;
      p.center = c;
      return std::unique_ptr<Strategy>(new Glft(p));
    };
  };
  auto as = [&](double g) { return std::unique_ptr<Strategy>(new AvellanedaStoikov(g, sp.fill_k, 10)); };
  double guard_ticks = 1;
  auto hmm = [&](double g) {
    GlftParams p;
    p.gamma = g;
    p.A = sp.fill_A;
    p.k = sp.fill_k;
    HmmGuardParams gp;
    gp.guard_ticks = guard_ticks;
    return std::unique_ptr<Strategy>(new HmmGlft(p, gp));
  };

  std::vector<std::pair<std::string, std::unique_ptr<Strategy>>> strategies;
  strategies.push_back({"", std::make_unique<FixedSpread>()});
  double g;
  g = tune("Avellaneda-Stoikov", as);
  strategies.push_back({"gamma=" + std::to_string(g).substr(0, 4), as(g)});
  for (Center c : {Center::Mid, Center::MicroPlain, Center::MicroHmm}) {
    const char* fam = c == Center::Mid ? "GLFT (mid)" : c == Center::MicroPlain ? "GLFT (Stoikov micro)" : "GLFT (HMM micro)";
    g = tune(fam, glft(c));
    strategies.push_back({"gamma=" + std::to_string(g).substr(0, 4), glft(c)(g)});
  }
  // HMM strategy: tune gamma and the adverse-selection guard (0 = off, 1, 3 ticks) jointly on train,
  // under the same participation constraint.
  double best_pnl = -1e300, hmm_gamma = gammas[0], best_guard = 0;
  for (double gt : {0.0, 1.0, 3.0}) {
    guard_ticks = gt;
    double gsel = tune("HMM-GLFT", hmm, gt);
    auto s = hmm(gsel);
    Metrics tr = runOne(train, sp, cfg, *s).m;
    double score = tr.fills >= min_fills ? tr.pnl : -1e200 + tr.fills;
    if (score > best_pnl) {
      best_pnl = score;
      hmm_gamma = gsel;
      best_guard = gt;
    }
  }
  guard_ticks = best_guard;
  strategies.push_back({"gamma=" + std::to_string(hmm_gamma).substr(0, 4) + ", guard=" +
                            std::to_string(static_cast<int>(best_guard)),
                        hmm(hmm_gamma)});
  if (a.count("rl")) {
    RlConfig rc;
    rc.max_lots = static_cast<int>(cfg.max_position);
    auto agent = std::make_unique<RlAgent>(rc);
    if (agent->load(a.at("rl"))) {
      agent->setLearning(false);
      strategies.push_back({"greedy policy", std::move(agent)});
    } else {
      std::fprintf(stderr, "could not load RL table %s\n", a.at("rl").c_str());
    }
  }

  std::unique_ptr<std::ofstream> csv;
  if (a.count("csv")) {
    csv = std::make_unique<std::ofstream>(a.at("csv"));
    *csv << "symbol,strategy,params,pnl,pnl_per_hour,sharpe_daily,max_drawdown,fills,volume,spread_capture,"
            "inventory_pnl,fees,mean_abs_position,time_at_limit,markout_1s,markout_10s,markout_60s,quote_updates,lot,tick\n";
  }
  std::printf("TEST split (tuned on train at >= %.0f%% of the fixed-spread baseline's train fills, i.e. >= %ld fills):\n",
              100 * min_part, min_fills);
  printHeader();
  std::vector<Run> runs;
  for (auto& [params, s] : strategies) {
    RlAgent* rl = dynamic_cast<RlAgent*>(s.get());
    EngineConfig c = cfg;
    if (rl) c.decision_ms = 1000;  // the agent acts once per second (as trained)
    Run r = runOne(test, sp, c, *s);
    if (!params.empty()) r.name += " [" + params + "]";
    printRow(r);
    runs.push_back(r);
    if (csv) {
      const Metrics& m = r.m;
      *csv << sym << ",\"" << s->name() << "\",\"" << params << "\"," << m.pnl << "," << m.pnl_per_hour << ","
           << m.sharpe_hourly << "," << m.max_drawdown << "," << m.fills << "," << m.volume << "," << m.spread_capture
           << "," << m.inventory_pnl << "," << m.fees << "," << m.mean_abs_position << "," << m.time_at_limit << ","
           << m.markout_mean[0] << "," << m.markout_mean[1] << "," << m.markout_mean[2] << "," << m.quote_updates << ","
           << cfg.lot << "," << tick << "\n";
    }
  }

  // Adverse selection by regime (fills of the mid-centred GLFT, which ignores the regime).
  std::printf("\nAdverse selection by HMM regime at fill time (GLFT-mid fills, mean 10s markout in ticks; "
              "negative = filled before the price moved against us):\n");
  const Run& base = runs[2];
  for (int k = 0; k < sp.K; ++k)
    std::printf("  regime %d: %6ld fills, markout10 %+.3f ticks\n", k, base.regime_fills[k], base.regime_markout10[k]);

  if (a.count("sensitivity")) {
    std::printf("\nSensitivity of HMM-GLFT [gamma=%.2f] on TEST:\n", hmm_gamma);
    printHeader();
    for (const char* q : {"risk-averse", "proportional", "power2", "power3"}) {
      EngineConfig c = cfg;
      c.queue = parseQueueModel(q);
      auto s = hmm(hmm_gamma);
      Run r = runOne(test, sp, c, *s);
      r.name = std::string("queue model ") + q;
      printRow(r);
    }
    for (double lat : {0.0, 20.0, 100.0, 300.0}) {
      EngineConfig c = cfg;
      c.latency_ms = lat;
      auto s = hmm(hmm_gamma);
      Run r = runOne(test, sp, c, *s);
      r.name = "latency " + std::to_string(static_cast<int>(lat)) + " ms";
      printRow(r);
    }
    for (double fee : {-0.00005, 0.0, 0.0002}) {
      EngineConfig c = cfg;
      c.maker_fee = fee;
      auto s = hmm(hmm_gamma);
      Run r = runOne(test, sp, c, *s);
      char buf[64];
      std::snprintf(buf, sizeof buf, "maker fee %+.3f%%", fee * 100);
      r.name = buf;
      printRow(r);
    }
  }
  return 0;
}
