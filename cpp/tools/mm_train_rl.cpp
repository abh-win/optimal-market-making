// mm_train_rl: train the SARSA(lambda) market maker on the TRAIN split.
//
//   mm_train_rl --data data/processed --symbol DOGEUSDT [--episodes 3000] [--episode-min 30]
//               [--out models/DOGEUSDT.rl.bin] [--lot-usd 100] [--max-pos 5]
//
// Each episode replays a random 30-minute window (plus 2 minutes of signal
// warm-up) with the agent learning online; epsilon decays linearly from 0.3 to 0.02.

#include <cmath>
#include <cstdio>
#include <random>

#include "common.h"
#include "mm/engine.h"
#include "mm/rl.h"

using namespace mm;

int main(int argc, char** argv) {
  auto a = cli::args(argc, argv);
  std::string dir = cli::get(a, "data", "data/processed"), sym = cli::get(a, "symbol", "DOGEUSDT");
  std::string model = cli::get(a, "model", "models/" + sym + ".model.txt");
  std::string out = cli::get(a, "out", "models/" + sym + ".rl.bin");
  int episodes = std::stoi(cli::get(a, "episodes", "3000"));
  double ep_ms = std::stod(cli::get(a, "episode-min", "30")) * 60e3;
  double tick = cli::jsonNumber(dir + "/" + sym + ".meta.json", "tick");
  SignalParams sp = SignalParams::load(model);
  MarketData train = MarketData::load(dir + "/" + sym + ".train", tick);

  EngineConfig cfg;
  cfg.lot = std::max(1.0, std::round(std::stod(cli::get(a, "lot-usd", "100")) / train.depth.front().mid()));
  cfg.max_position = std::stod(cli::get(a, "max-pos", "5"));
  cfg.latency_ms = std::stod(cli::get(a, "latency", "20"));
  cfg.queue = parseQueueModel(cli::get(a, "queue", "power2"));
  cfg.decision_ms = 1000;

  RlConfig rc;
  rc.max_lots = static_cast<int>(cfg.max_position);
  rc.vol_lo = 0.6 * sp.sigma_ticks;
  rc.vol_hi = 1.4 * sp.sigma_ticks;
  RlAgent agent(rc);

  const double t0 = train.depth.front().ts, t1 = train.depth.back().ts;
  const double warm = rc.warmup_s * 1000;
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> start(t0, t1 - ep_ms - warm);
  double avg_r = 0, avg_pnl = 0;
  for (int ep = 1; ep <= episodes; ++ep) {
    double eps = 0.3 + (0.02 - 0.3) * std::min(1.0, ep / (0.8 * episodes));
    agent.setEpsilon(eps);
    agent.setLearning(true);
    double s = start(rng);
    MarketData win = train.slice(s, s + warm + ep_ms);
    Engine e(win, sp, cfg);
    e.run(agent);
    avg_r += agent.episodeReward();
    avg_pnl += e.state().equity();
    if (ep % 200 == 0) {
      std::printf("episode %5d  eps %.3f  mean reward/ep %9.1f  mean PnL/ep %8.3f  coverage %.1f%%\n", ep, eps,
                  avg_r / 200, avg_pnl / 200, 100 * agent.coverage());
      std::fflush(stdout);
      avg_r = avg_pnl = 0;
    }
  }
  if (!agent.save(out)) {
    std::fprintf(stderr, "failed to save %s\n", out.c_str());
    return 1;
  }
  std::printf("saved %s\n", out.c_str());
  return 0;
}
