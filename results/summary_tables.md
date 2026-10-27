### DOGEUSDT

**naive cancel/replace** (test split)

| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |
|---|---|---|---|---|---|---|---|
| fixed-spread (join best) |  | -12.27 | 1310 | -0.748 | 2.89 | -15.17 | -0.69 / -0.86 / -0.90 |
| Avellaneda-Stoikov | gamma=0.01 | -1.14 | 279 | -0.327 | 0.33 | -1.48 | -0.83 / -1.05 / -1.03 |
| GLFT (mid) | gamma=0.00 | -2.07 | 262 | -0.631 | 0.32 | -2.39 | -0.83 / -1.06 / -1.20 |
| GLFT (Stoikov micro-price) | gamma=0.00 | -0.22 | 136 | -0.132 | 0.37 | -0.59 | -0.93 / -1.17 / -0.63 |
| GLFT (HMM micro-price) | gamma=0.00 | -0.25 | 134 | -0.149 | 0.38 | -0.63 | -0.96 / -1.18 / -0.52 |
| HMM-GLFT (micro-price + alpha + AS guard) | gamma=0.00, guard=0 | +0.16 | 145 | +0.091 | 0.10 | +0.07 | -0.95 / -1.22 / -1.00 |
| RL (SARSA-lambda) on HMM micro-price | greedy policy | -1.60 | 241 | -0.531 | 0.25 | -1.86 | -0.67 / -0.70 / -0.32 |

**queue-priority retention (keep ≤ 2 ticks)** (test split)

| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |
|---|---|---|---|---|---|---|---|
| fixed-spread (join best) |  | -6.62 | 931 | -0.568 | 2.08 | -8.70 | -0.69 / -0.90 / -0.68 |
| Avellaneda-Stoikov | gamma=0.01 | +0.38 | 184 | +0.165 | 0.55 | -0.17 | -0.81 / -1.09 / -0.79 |
| GLFT (mid) | gamma=0.00 | +2.12 | 185 | +0.913 | 0.58 | +1.54 | -0.67 / -0.96 / -0.78 |
| GLFT (Stoikov micro-price) | gamma=0.00 | +2.49 | 91 | +2.186 | 0.27 | +2.22 | -0.86 / -1.06 / -0.30 |
| GLFT (HMM micro-price) | gamma=0.00 | +1.37 | 93 | +1.177 | 0.18 | +1.19 | -1.05 / -1.25 / -1.37 |
| HMM-GLFT (micro-price + alpha + AS guard) | gamma=0.00, guard=0 | +0.49 | 82 | +0.481 | -0.17 | +0.66 | -1.28 / -1.67 / -1.41 |
| RL (SARSA-lambda) on HMM micro-price | greedy policy | -0.89 | 152 | -0.470 | 0.38 | -1.28 | -0.61 / -0.88 / -0.57 |

### ADAUSDT

**naive cancel/replace** (test split)

| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |
|---|---|---|---|---|---|---|---|
| fixed-spread (join best) |  | -5.15 | 852 | -0.134 | 4.05 | -9.20 | -0.33 / -0.22 / -0.17 |
| Avellaneda-Stoikov | gamma=0.10 | +0.45 | 8 | +1.250 | -0.41 | +0.85 | -2.00 / -2.25 / -2.50 |
| GLFT (mid) | gamma=0.10 | -0.41 | 10 | -0.900 | -0.36 | -0.04 | -1.70 / -1.70 / -2.10 |
| GLFT (Stoikov micro-price) | gamma=0.00 | +0.83 | 12 | +1.542 | -0.43 | +1.26 | -1.50 / -1.67 / -1.67 |
| GLFT (HMM micro-price) | gamma=0.00 | +0.83 | 12 | +1.542 | -0.43 | +1.26 | -1.50 / -1.67 / -1.67 |
| HMM-GLFT (micro-price + alpha + AS guard) | gamma=0.00, guard=0 | +0.83 | 12 | +1.542 | -0.43 | +1.26 | -1.50 / -1.67 / -1.67 |
| RL (SARSA-lambda) on HMM micro-price | greedy policy | -1.08 | 87 | -0.276 | -0.88 | -0.20 | -0.63 / -0.48 / -0.36 |

**queue-priority retention (keep ≤ 2 ticks)** (test split)

| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |
|---|---|---|---|---|---|---|---|
| fixed-spread (join best) |  | -3.43 | 216 | -0.353 | 1.71 | -5.14 | -0.37 / -0.25 / -0.29 |
| Avellaneda-Stoikov | gamma=0.10 | +0.61 | 7 | +1.929 | -0.16 | +0.77 | -1.64 / -1.93 / -2.36 |
| GLFT (mid) | gamma=0.10 | +0.58 | 8 | +1.625 | -0.14 | +0.72 | -1.50 / -1.88 / -1.88 |
| GLFT (Stoikov micro-price) | gamma=0.01 | +0.52 | 7 | +1.643 | -0.25 | +0.77 | -1.93 / -2.21 / -2.64 |
| GLFT (HMM micro-price) | gamma=0.01 | +0.52 | 7 | +1.643 | -0.25 | +0.77 | -1.93 / -2.21 / -2.64 |
| HMM-GLFT (micro-price + alpha + AS guard) | gamma=0.00, guard=0 | +0.61 | 5 | +2.700 | -0.38 | +0.99 | -2.70 / -3.10 / -2.90 |
| RL (SARSA-lambda) on HMM micro-price | greedy policy | -0.32 | 17 | -0.412 | -0.49 | +0.18 | -0.97 / -1.21 / -1.38 |

### XRPUSDT

**naive cancel/replace** (test split)

| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |
|---|---|---|---|---|---|---|---|
| fixed-spread (join best) |  | -13.55 | 1575 | -1.117 | 1.93 | -15.48 | -0.91 / -1.28 / -1.74 |
| Avellaneda-Stoikov | gamma=0.00 | +0.11 | 177 | +0.080 | 0.31 | -0.20 | -1.31 / -2.05 / -3.37 |
| GLFT (mid) | gamma=0.00 | -1.03 | 225 | -0.593 | 0.39 | -1.42 | -1.26 / -1.80 / -2.94 |
| GLFT (Stoikov micro-price) | gamma=0.00 | +0.93 | 208 | +0.580 | 0.20 | +0.73 | -1.34 / -2.25 / -2.33 |
| GLFT (HMM micro-price) | gamma=0.00 | +1.03 | 205 | +0.652 | 0.20 | +0.83 | -1.34 / -2.23 / -2.33 |
| HMM-GLFT (micro-price + alpha + AS guard) | gamma=0.00, guard=3 | -1.77 | 207 | -1.109 | -0.03 | -1.74 | -1.41 / -2.03 / -3.08 |
| RL (SARSA-lambda) on HMM micro-price | greedy policy | -5.08 | 504 | -1.309 | -0.30 | -4.77 | -1.23 / -1.49 / -3.02 |

**queue-priority retention (keep ≤ 2 ticks)** (test split)

| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |
|---|---|---|---|---|---|---|---|
| fixed-spread (join best) |  | -9.95 | 1245 | -1.038 | 1.50 | -11.45 | -0.93 / -1.33 / -2.07 |
| Avellaneda-Stoikov | gamma=0.00 | +0.90 | 141 | +0.830 | 0.28 | +0.62 | -1.32 / -2.47 / -3.31 |
| GLFT (mid) | gamma=0.00 | -0.55 | 209 | -0.342 | 0.31 | -0.87 | -1.22 / -1.94 / -3.21 |
| GLFT (Stoikov micro-price) | gamma=0.00 | +1.50 | 166 | +1.172 | 0.23 | +1.26 | -1.33 / -2.39 / -2.09 |
| GLFT (HMM micro-price) | gamma=0.00 | +1.50 | 165 | +1.179 | 0.23 | +1.26 | -1.33 / -2.38 / -2.08 |
| HMM-GLFT (micro-price + alpha + AS guard) | gamma=0.00, guard=3 | +0.43 | 164 | +0.337 | 0.01 | +0.42 | -1.51 / -2.55 / -4.16 |
| RL (SARSA-lambda) on HMM micro-price | greedy policy | -3.02 | 402 | -0.976 | -0.07 | -2.95 | -1.14 / -1.50 / -2.01 |

### TRXUSDT

**naive cancel/replace** (test split)

| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |
|---|---|---|---|---|---|---|---|
| fixed-spread (join best) |  | -1.46 | 467 | -1.042 | 0.33 | -1.78 | -1.47 / -1.61 / -2.31 |
| Avellaneda-Stoikov | gamma=1.00 | -0.99 | 180 | -1.844 | 0.27 | -1.26 | -2.09 / -2.47 / -4.54 |
| GLFT (mid) | gamma=0.30 | -0.71 | 123 | -1.939 | 0.26 | -0.98 | -2.43 / -2.70 / -4.14 |
| GLFT (Stoikov micro-price) | gamma=0.30 | -0.55 | 121 | -1.525 | 0.24 | -0.79 | -2.81 / -2.84 / -2.93 |
| GLFT (HMM micro-price) | gamma=0.30 | -0.55 | 121 | -1.525 | 0.24 | -0.79 | -2.81 / -2.84 / -2.93 |
| HMM-GLFT (micro-price + alpha + AS guard) | gamma=0.00, guard=0 | -0.20 | 130 | -0.527 | 0.28 | -0.48 | -3.01 / -3.82 / -4.27 |
| RL (SARSA-lambda) on HMM micro-price | greedy policy | -0.12 | 103 | -0.379 | 0.18 | -0.30 | -3.32 / -3.53 / -3.27 |

**queue-priority retention (keep ≤ 2 ticks)** (test split)

| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |
|---|---|---|---|---|---|---|---|
| fixed-spread (join best) |  | -1.25 | 398 | -1.051 | 0.30 | -1.56 | -1.77 / -1.93 / -2.67 |
| Avellaneda-Stoikov | gamma=3.00 | -0.63 | 120 | -1.767 | 0.14 | -0.77 | -2.39 / -2.48 / -5.35 |
| GLFT (mid) | gamma=0.30 | -0.54 | 114 | -1.580 | 0.22 | -0.76 | -3.14 / -3.20 / -4.36 |
| GLFT (Stoikov micro-price) | gamma=0.30 | -0.46 | 114 | -1.349 | 0.21 | -0.67 | -2.76 / -2.83 / -4.43 |
| GLFT (HMM micro-price) | gamma=0.30 | -0.46 | 114 | -1.349 | 0.21 | -0.67 | -2.76 / -2.83 / -4.43 |
| HMM-GLFT (micro-price + alpha + AS guard) | gamma=0.03, guard=0 | -0.39 | 98 | -1.342 | 0.31 | -0.71 | -3.38 / -3.98 / -4.55 |
| RL (SARSA-lambda) on HMM micro-price | greedy policy | -0.01 | 84 | -0.043 | 0.17 | -0.18 | -3.49 / -3.77 / -3.17 |

