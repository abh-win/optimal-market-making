#!/usr/bin/env bash
# Full pipeline on the recorded data:
#   research models (Python) -> C++ build -> RL training -> backtests (tuned on train, reported on test)
# Usage: scripts/run_all.sh [SYMBOLS] [RL_EPISODES]
set -euo pipefail
cd "$(dirname "$0")/.."
SYMS=${1:-DOGEUSDT,ADAUSDT,XRPUSDT,TRXUSDT}
EPISODES=${2:-3000}
mkdir -p results models
.venv/bin/python research/fit_models.py --symbols "$SYMS" | tee results/research.txt
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build cpp/build -j > /dev/null
: > results/backtest.txt
for S in ${SYMS//,/ }; do
  ./cpp/build/mm_train_rl --symbol "$S" --episodes "$EPISODES" | tee "results/$S/rl_training.txt"
  # naive requoting (cancel/replace whenever the target moves) vs queue-priority-aware (keep orders up to 2 ticks passive)
  ./cpp/build/mm_backtest --symbol "$S" --rl "models/$S.rl.bin" --keep-ticks 0 --csv "results/$S/backtest_keep0.csv" --frontier "results/$S/frontier_keep0.csv" \
    | tee -a results/backtest.txt
  ./cpp/build/mm_backtest --symbol "$S" --rl "models/$S.rl.bin" --keep-ticks 2 --sensitivity \
    --csv "results/$S/backtest_keep2.csv" --frontier "results/$S/frontier_keep2.csv" | tee -a results/backtest.txt
  echo | tee -a results/backtest.txt
done
.venv/bin/python research/summarize_backtests.py --symbols "$SYMS" > /dev/null
