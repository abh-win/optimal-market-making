# Optimal Market Making via Latent-State Micro-Price Estimation

An end-to-end market-making research stack on **real Binance USD-M futures order-book data** that
this project recorded itself (10 levels, 100 ms snapshots, plus every trade). It has three parts:

1. **HMM micro-price.** Stoikov's micro-price is extended with a hidden Markov model whose regimes
   (sell pressure / neutral / buy pressure) are inferred from L1 and L2–L5 order-book imbalances,
   multi-level order-flow imbalance and trade flow. The regimes classify adverse selection.
2. **C++ tick-level backtester.** It tracks queue position for every resting order: FIFO trade
   depletion, three cancellation models, latency on entries and cancels, stale-quote fills while a
   cancel is in flight, trade-through and cross fills. Adverse-selection costs are measured with
   markouts.
3. **Strategy ladder.** Fixed spread → Avellaneda–Stoikov → Guéant–Lehalle–Fernandez-Tapia (GLFT)
   closed-form quotes with inventory limits, centred on the mid, the Stoikov micro-price or the HMM
   micro-price → HMM-GLFT with a regime alpha and an adverse-selection guard → a **SARSA(λ)
   reinforcement-learning agent** that sets bid/ask skew under strict inventory limits.

The specification and the papers anchoring each part are listed in `SPEC.md`. The earlier
micro-price v1 study in `~/Projects/Market Making` is left untouched; this repository
re-implements and extends it.

## Data

`recorder/record_binance.py` subscribes to the public `depth20@100ms` and `trade` streams and
writes hourly gzip CSVs. Four large-tick perpetuals were recorded:

| symbol | tick | tick / price |
|---|---|---|
| DOGEUSDT | 0.00001 | ≈ 1.25 bp |
| ADAUSDT | 0.0001 | ≈ 4.5 bp |
| XRPUSDT | 0.0001 | ≈ 0.8 bp |
| TRXUSDT | 0.00001 | ≈ 0.3 bp |

All four have a median spread of one tick, where queue position and micro-price matter most.
`research/mmr/prepare.py` converts the recordings into:

- compact binaries for the C++ engine, split chronologically: the first 65% of time for training,
  the last 35% for testing;
- a 1-second feature table for the research models.

Recorded on 2026-10-17/18 from 19:11 to 09:10 UTC, giving **13.85 hours per symbol**:

| symbol | depth snapshots | trades | train / test |
|---|---|---|---|
| DOGEUSDT | 394,333 | 461,634 | 9.0 h / 4.85 h |
| ADAUSDT | 381,087 | 210,517 | 9.0 h / 4.85 h |
| XRPUSDT | 454,682 | 648,101 | 9.0 h / 4.85 h |
| TRXUSDT | 231,518 | 71,964 | 9.0 h / 4.85 h |

The recorder reconnects automatically. Binance drops idle websockets, and a few gaps of about 4
seconds occurred; the features forward-fill across them.

## Research models (`research/`)

**Features**, one row per second; the C++ engine reproduces them exactly:

- `imb1`: L1 imbalance, q_b / (q_b + q_a).
- `imb_deep`: depth imbalance of levels 2–5.
- `ofi1` / `ofi_deep`: Cont–Kukanov–Stoikov order-flow imbalance at level 1 and summed over levels
  2–5 (multi-level OFI), normalised by depth.
- `tflow5`: 5-second signed trade-volume share.

**HMM** (`mmr/hmm.py`, written from scratch). It is a Gaussian HMM with diagonal covariances,
fitted by Baum–Welch with scaled forward–backward recursions:

- **Training data:** the training split plus its **bid/ask mirror image**, since exchanging bid and
  ask negates every signed feature.
- **Symmetry by construction:** the EM is constrained to the mirror-symmetric parameter space:
  means (−m, 0, +m), tied variances, and A = (A + PAP)/2. The states are therefore exactly
  *sell pressure / neutral / buy pressure*. An unconstrained fit converged to an asymmetric local
  optimum on XRP, which broke the symmetrised micro-price.
- **Inference:** trading uses only the **causal filtered** posterior P(z_t | x_1..t), never
  Viterbi.

**Micro-price** (`mmr/microprice.py`). This is Stoikov (2018), estimated in his notation:

- Q, R and T come from one-step transitions on states x = (imbalance bucket, spread).
- G¹ = (I−Q)⁻¹RK, B = (I−Q)⁻¹T and G* = Σ Bⁱ G¹.
- The training data is symmetrised (I → 1−I, ΔM → −ΔM).

**Extension:** x = (imbalance bucket, spread, regime). The regime is mirrored along with the
imbalance. Rare cells (fewer than 50 observations) become absorbing, and regime-specific
adjustments are shrunk toward the pooled Stoikov value of the same cell with weight n / (n + 200).

**Also estimated:**

- **Regime drift:** E[ΔM over the next second | regime].
- **Fill intensity:** λ(δ) = A·e^(−kδ), from how far beyond the mid market orders trade each second.
- **Volatility:** σ of 1-second mid changes.

**Adverse selection by regime:** each trade's aggressor-signed 10-second price impact. Averaging
the buy-side and sell-side means cancels a common drift and leaves the information component.

## Backtester (`cpp/`)

| piece | behaviour |
|---|---|
| Event loop | Trades and depth snapshots are replayed in exchange-time order. Trades stamped at or before a snapshot are applied first, because the snapshot already reflects them. |
| Entry, cancel, latency | Quotes take effect `latency_ms` after the decision (default 20 ms). A replaced order **stays fillable until its cancel lands**, so stale quotes can get picked off. |
| Queue position | A new order joins the **back** of the visible queue at its price. Trades at that price consume the queue ahead first (FIFO); volume beyond it fills us. A trade *through* our price, or the opposite best quote crossing it, fills us fully. |
| Cancellations | Depth decreases at our price that trades don't explain are cancellations. How many were ahead of us depends on the model: `risk-averse` (none), `proportional`, or the power-law probability models `power2`/`power3`: fraction ahead f(x) = xⁿ / (xⁿ + (1−x)ⁿ), with x = queue ahead / depth. |
| Queue priority retention | `--keep-ticks N` keeps a resting order instead of cancel/replace while it sits at most N ticks on the passive side of the new target. On one-tick books, giving up queue priority is the main source of adverse selection. |
| Risk | A hard inventory cap: quotes that could breach the limit are never sent. Maker fee or rebate is configurable. |
| Accounting | Mark-to-mid equity and PnL split into spread capture (fill price vs mid at fill), inventory PnL and fees. Markouts at 1 s, 10 s and 60 s, overall and by regime. Daily Sharpe from 15-minute PnL. Max drawdown. Time at the inventory limit. |

The online signals (`signals.cpp`) reproduce the Python per-second features and HMM posteriors
exactly: `research/check_parity.py` shows a maximum feature difference of 5e-9 and a maximum
posterior difference of 5e-9, with 100% regime agreement.

### Strategies (`cpp/src/strategies.cpp`, `cpp/src/rl.cpp`)

| strategy | quotes |
|---|---|
| Fixed spread | join the best bid/ask; step back one tick on the side that would increase inventory |
| Avellaneda–Stoikov (2008) | reservation price r = c − qγσ²τ, spread γσ²τ + (2/γ)·ln(1 + γ/k) |
| GLFT (2013) | δ_b(q) = c₁ + (2q+1)/2·σc₂ and δ_a(q) = c₁ − (2q−1)/2·σc₂, where c₁ = ln(1+γ/k)/γ and c₂ = √(γ/(2Ak)·(1+γ/k)^(1+k/γ)). Centred on the **mid**, the **Stoikov micro-price** or the **HMM micro-price**. |
| HMM-GLFT | GLFT around the HMM micro-price, plus a Cartea–Jaimungal-style alpha skew from the regime drift, plus an **adverse-selection guard** that widens the side informed flow is hitting when P(regime) > 0.6 |
| RL agent (Spooner et al. 2018) | Tabular SARSA(λ) with replacing traces. **State:** inventory (−5…5 lots) × regime × micro-price deviation × spread × volatility bucket. **Actions:** half-spread ∈ {0.5, 1, 2} ticks × skew ∈ {−0.5, 0, +0.5} ticks around the HMM micro-price, or pull both quotes. **Reward:** asymmetrically dampened PnL, ΔPnL − η·max(0, qΔM) − κq². Trained on random 30-minute windows of the training split with decaying ε-greedy exploration. |

Every strategy parameter (γ, and for HMM-GLFT also the guard size ∈ {0, 1, 3} ticks) is **tuned on
the training split by PnL**. All reported numbers come from the untouched test split.

## Run

```bash
python -m venv .venv && .venv/bin/pip install websockets numpy pandas pyarrow scipy scikit-learn matplotlib pytest
.venv/bin/python recorder/record_binance.py --hours 14          # record (or point at your own data/raw)
scripts/run_all.sh                                               # research fit + RL training + backtests
.venv/bin/python research/check_parity.py DOGEUSDT --split test  # C++ vs Python signal parity
.venv/bin/python -m pytest -q research/tests && cpp/build/mm_tests
```

## Results

All numbers below are on the **test split**, the last 4.85 hours, never used for fitting or
tuning. Lots are about $100, the inventory cap is 5 lots, latency is 20 ms, the queue model is
`power2`, and the maker fee is 0 unless stated. Full outputs are in `results/research.txt`,
`results/backtest.txt`, `results/summary_tables.md`, and per symbol in
`results/<SYM>/{research.png, frontier.png, *.csv}`.

### 1. Micro-price and regimes (out of sample)

The table gives forecast skill for the mid price after the next mid change (Stoikov's criterion):
1 − MSE / MSE(mid).

| | DOGE | ADA | XRP | TRX |
|---|---|---|---|---|
| weighted mid | 0.0214 | **0.0763** | 0.0074 | **0.0318** |
| Stoikov micro-price | 0.0214 | 0.0736 | 0.0099 | 0.0302 |
| **HMM micro-price** (filtered regime) | 0.0222 | 0.0733 | 0.0100 | 0.0307 |
| linear L1–L5 imbalance (OLS) | **0.0361** | 0.0758 | **0.0150** | 0.0299 |

- **HMM vs Stoikov:** the regime-augmented micro-price matches or slightly beats plain Stoikov on
  three of four books (DOGE +4%, XRP +1%, TRX +2%; ADA −0.4%). The gains are small: once the L1
  imbalance is known, the regime adds little to the fair-value estimate itself.
- **Deep-book imbalance:** a plain linear model on cumulative L1–L5 imbalances is the best
  forecaster on DOGE and XRP. Information beyond level 1 is real, which supports the L2–L5 inputs.
- **Directional pressure:** the regimes classify it clearly. Expected mid drift over the next
  second is about ±0.2–0.3 ticks in the sell/buy-pressure regimes (DOGE −0.22 / +0.19,
  XRP −0.31 / +0.29) against about 0 in the neutral regime, which holds 85–91% of the time.
- **Adverse selection:** on DOGE and XRP, aggressors trading *with* the regime are more informed.
  Mean 10-second price impact of buyer-initiated trades:

  | | buy-pressure regime | neutral regime |
  |---|---|---|
  | DOGE | 0.84 ticks | 0.37 ticks |
  | XRP | 0.95 ticks | 0.59 ticks |

  These are the fills that pick off a maker's asks. On TRX and ADA the effect is absent or
  reversed.

### 2. Strategies (test PnL in $, fills, PnL per fill in ticks)

With naive cancel/replace:

| strategy | DOGE | ADA | XRP | TRX |
|---|---|---|---|---|
| fixed spread (join best) | −12.27 (1310, −0.75) | −5.15 (852, −0.13) | −13.55 (1575, −1.12) | −1.46 (467, −1.04) |
| Avellaneda–Stoikov | −1.14 (279) | +0.45 (8) | +0.11 (177) | −0.99 (180) |
| GLFT (mid) | −2.07 (262, −0.63) | −0.41 (10) | −1.03 (225, −0.59) | −0.71 (123, −1.94) |
| GLFT (Stoikov micro-price) | −0.22 (136, −0.13) | +0.83 (12) | +0.93 (208, +0.58) | −0.55 (121, −1.53) |
| GLFT (HMM micro-price) | −0.25 (134, −0.15) | +0.83 (12) | **+1.03** (205, +0.65) | −0.55 (121, −1.53) |
| HMM-GLFT (alpha + guard) | **+0.16** (145, +0.09) | +0.83 (12) | −1.77 (207, −1.11) | −0.20 (130, −0.53) |
| RL agent (SARSA-λ) | −1.60 (241, −0.53) | −1.08 (87) | −5.08 (504, −1.31) | **−0.12** (103, −0.38) |

With queue-priority retention (`--keep-ticks 2`):

| strategy | DOGE | ADA | XRP | TRX |
|---|---|---|---|---|
| fixed spread (join best) | −6.62 (931) | −3.43 (216) | −9.95 (1245) | −1.25 (398) |
| Avellaneda–Stoikov | +0.38 (184) | +0.61 (7) | +0.90 (141) | −0.63 (120) |
| GLFT (mid) | +2.12 (185, +0.91) | +0.58 (8) | −0.55 (209, −0.34) | −0.54 (114) |
| GLFT (Stoikov micro-price) | **+2.49** (91, +2.19) | +0.52 (7) | **+1.50** (166, +1.17) | −0.46 (114) |
| GLFT (HMM micro-price) | +1.37 (93, +1.18) | +0.52 (7) | **+1.50** (165, +1.18) | −0.46 (114) |
| HMM-GLFT (alpha + guard) | +0.49 (82) | +0.61 (5) | +0.43 (164) | −0.39 (98) |
| RL agent (SARSA-λ) | −0.89 (152) | −0.32 (17) | −3.02 (402) | **−0.01** (84) |

**Findings, ordered by how robust they are:**

1. **Joining the touch and requoting on every move loses money on every book.** Each fill earns
   about half a tick of spread but suffers 0.7–1.3 ticks of 10-second adverse selection, because
   a back-of-queue order is filled exactly when its level is about to be swept.
2. **Queue priority is the largest single lever.** Keeping resting orders up to 2 ticks behind the
   target, instead of cancel/replacing, improves almost every strategy on every symbol. Fixed
   spread on DOGE goes from −12.3 to −6.6, and GLFT-mid from −2.1 to +2.1.
3. **Centring on a micro-price beats centring on the mid on all four books, in both modes.**
   Examples: XRP −1.03 → +0.93 (naive) and −0.55 → +1.50 (priority); DOGE −2.07 → −0.22 (naive).
   The Stoikov and HMM micro-prices are essentially tied in PnL, consistent with section 1.
4. **The HMM alpha skew and adverse-selection guard help on some books and hurt on others.** In
   naive mode HMM-GLFT is the best strategy on DOGE (+0.16, the only positive one) and second-best
   on TRX, but it costs about $2.8 on XRP. The guard size chosen on train (0, 1 or 3 ticks) did not
   transfer reliably to test.
5. **The RL agent learns the inventory constraint but not a better quoting policy.** It reaches
   ~50% state–action coverage over 3,000 training episodes, keeps |q| well inside the limit, and
   is the best (least-losing) strategy on TRX, the thinnest book. Elsewhere it trades too often
   and underperforms GLFT-micro. A tabular agent on 9 hours of data is sample-starved.
6. **Fees decide profitability.** For HMM-GLFT, a 0.5 bp rebate adds $0.02 (ADA) to $0.61
   (XRP) over the test window, while a 2 bp maker fee (Binance's retail tier) turns every
   configuration negative. Latency (0–300 ms) and the cancellation model move HMM-GLFT's PnL by up
   to about $1.4 (DOGE). That is the same order as the gaps between GLFT variants, which is
   another sign those gaps are within noise. The `risk-averse` model gives the fewest fills, as
   expected.

**Caveat on significance.** The model-based strategies make only about 100–200 fills in 4.85 hours,
so the PnL differences between GLFT variants are of the order of the noise; per-fill markout
standard deviations are about 1–2 ticks. The robust conclusions are 1–3. The gamma frontiers
(`results/<SYM>/frontier.png`) show the whole PnL-vs-volume trade-off rather than one tuned point.

**Tuning rule.** Each strategy takes the highest train PnL among settings that trade at least 25%
of the fixed-spread baseline's train volume. Without that constraint, the tuner simply picks
"don't quote", because every quoting setting loses on train. When no setting reaches the
threshold, as with GLFT on most books, the most active setting is used.


## Layout

```
recorder/          record_binance.py (websocket -> hourly gzip CSV)
research/mmr/      prepare.py  hmm.py  microprice.py
research/          fit_models.py  check_parity.py  tests/
cpp/include/mm/    data.h  signals.h  engine.h  strategies.h  rl.h
cpp/src/           data.cpp  signals.cpp  engine.cpp  strategies.cpp  rl.cpp
cpp/tools/         mm_backtest.cpp  mm_train_rl.cpp  mm_dump_signals.cpp
cpp/tests/         queue/fill/latency/stale-quote/limit/fee tests, GLFT properties
models/            exported research models (<SYM>.model.txt) and RL Q-tables
results/           research.txt, backtest.txt, per-symbol CSV/PNG/JSON
```
