#!/usr/bin/env python3
"""Fit the research models for each recorded symbol and export them for the C++ backtester.

  fit_models.py [--symbols DOGEUSDT,ADAUSDT,XRPUSDT,TRXUSDT] [--raw data/raw] [--out data/processed]

Per symbol:
 1. prepare: raw recordings -> train/test binaries (first 65% / last 35% of time) + 1-s features
 2. HMM: 3-state Gaussian HMM on [L1 imbalance, L2-5 imbalance, L1 OFI, L2-5 OFI, 5-s trade flow],
    fit on the train split plus its bid/ask mirror image; states sorted sell -> neutral -> buy
 3. causal filtered regimes for every second
 4. micro-prices: Stoikov G*(imbalance, spread) and the regime-augmented G*(imbalance, spread, regime),
    fit on train, evaluated on test (MSE of future mid at 1-60 s, vs mid / weighted mid / linear L1-5 model)
 5. adverse selection by regime: price impact of trades (aggressor-signed mid change after 10 s)
 6. fill intensity lambda(delta) = A exp(-k delta) and 1-s volatility, for GLFT / Avellaneda-Stoikov
 7. write models/<SYM>.model.txt (C++) and results/<SYM>/research.json + plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from mmr.hmm import GaussianHMM  # noqa: E402
from mmr.microprice import MicroPrice  # noqa: E402
from mmr.prepare import TRADE_COLS, complete_gzip, prepare, read_csv_any  # noqa: E402

FEATS = ["imb1c", "imb_deep", "ofi1", "ofi_deep", "tflow5"]
HORIZONS = [1, 5, 10, 30, 60]


def hmm_matrix(f: pd.DataFrame) -> np.ndarray:
    return np.column_stack([2 * f.imb1 - 1, f.imb_deep, f.ofi1, f.ofi_deep, f.tflow5]).astype(float)


def fit_hmm(f: pd.DataFrame, K: int = 3, seed: int = 0):
    tr = f[f.split == "train"]
    X = hmm_matrix(tr)
    scale = np.sqrt(np.mean(X ** 2, axis=0))  # RMS (mean is ~0 by mirror symmetry)
    Z = X / scale
    # Symmetric 3-state HMM fit on the data plus its bid/ask mirror image.
    model = GaussianHMM(K, n_iter=200, n_init=4, seed=seed, symmetric=True).fit([Z, -Z])
    # Orient: state 0 = sell pressure (negative signed flow), state 2 = buy pressure.
    if model.means[0].sum() > model.means[2].sum():
        model.reorder(np.array([2, 1, 0]))
    mirror = np.array([2, 1, 0])
    return model, scale, mirror


def fill_intensity(f: pd.DataFrame, trades: pd.DataFrame, tick: float):
    """lambda(delta): rate (per second, per side) at which market orders trade at least
    delta ticks away from the mid prevailing at the start of the second."""
    mid = f.set_index("sec")["mid"]
    tsec = (trades.txn_ms // 1000).astype(np.int64)
    m0 = mid.reindex(tsec - 1).to_numpy()
    side = np.where(trades.buyer_is_maker == 1, -1.0, 1.0)
    dist = side * (trades.price.to_numpy() - m0) / tick
    df = pd.DataFrame({"sec": tsec, "side": side, "dist": dist}).dropna()
    mx = df.groupby(["sec", "side"])["dist"].max()
    n_sec = len(f)
    deltas = np.arange(0.5, 6.01, 0.5)
    lam = np.array([(mx >= d).sum() / (2 * n_sec) for d in deltas])
    ok = lam > 0
    slope, intercept = np.polyfit(deltas[ok], np.log(lam[ok]), 1)
    return float(np.exp(intercept)), float(-slope), deltas, lam


def evaluate_micro(f, tick, preds: dict[str, np.ndarray]):
    rows = []
    mid = f["mid"].to_numpy()
    # Stoikov's criterion: the mid price after the *next* mid change.
    nxt = np.full(len(mid), np.nan)
    j = len(mid)
    for i in range(len(mid) - 1, -1, -1):
        if i + 1 < len(mid) and mid[i + 1] != mid[i]:
            j = i + 1
        if j < len(mid):
            nxt[i] = mid[j]
    tgt = (nxt - mid) / tick
    okn = np.isfinite(tgt)
    base_n = np.mean(tgt[okn] ** 2)
    for name, p in preds.items():
        adj = (p - mid) / tick
        ok = okn & np.isfinite(adj)
        rows.append({"horizon_s": "next mid change", "model": name, "mse_ticks2": np.mean((tgt[ok] - adj[ok]) ** 2),
                     "skill_vs_mid": 1 - np.mean((tgt[ok] - adj[ok]) ** 2) / base_n,
                     "hit_rate": np.mean(np.sign(adj[ok & (tgt != 0)]) == np.sign(tgt[ok & (tgt != 0)]))})
    for h in HORIZONS:
        fut = np.roll(mid, -h)
        valid = np.arange(len(mid)) < len(mid) - h
        target = (fut - mid) / tick
        base = np.mean(target[valid] ** 2)
        for name, p in preds.items():
            adj = (p - mid) / tick
            ok = valid & np.isfinite(adj)
            mse = np.mean((target[ok] - adj[ok]) ** 2)
            moved = ok & (target != 0)
            hit = np.mean(np.sign(adj[moved]) == np.sign(target[moved])) if moved.any() else np.nan
            rows.append({"horizon_s": h, "model": name, "mse_ticks2": mse, "skill_vs_mid": 1 - mse / base,
                         "hit_rate": hit})
    return pd.DataFrame(rows)


def write_cpp_model(path, tick, hmm, scale, plain, micro, drift, A, k, sigma):
    with open(path, "w") as fh:
        w = lambda key, vals: fh.write(key + " " + " ".join(f"{v:.12g}" for v in np.ravel(vals)) + "\n")  # noqa: E731
        fh.write(f"tick {tick:.12g}\n")
        fh.write(f"hmm {hmm.K} {len(FEATS)}\n")
        w("feat_scale", scale)
        w("pi", hmm.pi)
        w("A", hmm.A)
        w("means", hmm.means)
        w("vars", hmm.vars)
        fh.write(f"micro {micro.n_imb} {micro.n_spread} {micro.n_reg}\n")
        w("gstar_plain", plain.table()[0])
        w("gstar_hmm", micro.table())
        w("drift", drift)
        fh.write(f"fill {A:.12g} {k:.12g}\n")
        fh.write(f"sigma {sigma:.12g}\n")


def run_symbol(sym, raw, out, results_dir, models_dir):
    meta = prepare(raw, sym, out)
    tick = meta["tick"]
    f = pd.read_parquet(os.path.join(out, f"{sym}.seconds.parquet"))
    print(f"\n=== {sym}: {meta['hours']} h recorded, tick {tick}, {meta['n_depth']} snapshots, {meta['n_trades']} trades")

    hmm, scale, mirror = fit_hmm(f)
    post = hmm.filter(hmm_matrix(f) / scale)  # causal, over the whole timeline
    reg = post.argmax(axis=1)
    f["regime"] = reg
    tr, te = f.split.to_numpy() == "train", f.split.to_numpy() == "test"

    # Regime-conditional drift: E[mid change over the next second | regime] (ticks), on train.
    dmid1 = (np.roll(f.mid.to_numpy(), -1) - f.mid.to_numpy()) / tick
    dmid1[-1] = 0
    drift = np.array([dmid1[tr & (reg == k)].mean() if (tr & (reg == k)).any() else 0.0 for k in range(hmm.K)])

    # Micro-prices.
    ftr = f[tr].reset_index(drop=True)
    plain = MicroPrice(n_imb=5, n_spread=2, n_regimes=1)
    plain.fit(plain.transitions(ftr, np.zeros(len(ftr), dtype=int), tick))
    micro = MicroPrice(n_imb=5, n_spread=2, n_regimes=hmm.K, mirror=mirror)
    micro.fit(micro.transitions(ftr, ftr.regime.to_numpy(), tick))

    fte = f[te].reset_index(drop=True)
    post_te = post[te]
    g_plain = plain.adjust(fte.imb1, fte.spread, np.zeros(len(fte), dtype=int))
    g_hmm_hard = micro.adjust(fte.imb1, fte.spread, fte.regime)
    g_hmm = sum(post_te[:, r] * micro.adjust(fte.imb1, fte.spread, np.full(len(fte), r)) for r in range(hmm.K))
    # Linear benchmark: OLS of the 10-s mid change on L1..L5 cumulative imbalances (train).
    cols = [f"imb_{j}" for j in range(1, 6)]
    ytr = (ftr.mid.shift(-10) - ftr.mid).to_numpy() / tick
    ok = np.isfinite(ytr)
    Xtr = np.column_stack([np.ones(ok.sum()), ftr.loc[ok, cols].to_numpy()])
    beta = np.linalg.lstsq(Xtr, ytr[ok], rcond=None)[0]
    lin = np.column_stack([np.ones(len(fte)), fte[cols].to_numpy()]) @ beta
    preds = {
        "mid": fte.mid.to_numpy(),
        "weighted mid": fte.wmid.to_numpy(),
        "Stoikov micro-price": fte.mid.to_numpy() + g_plain * tick,
        "HMM micro-price (filtered regime)": fte.mid.to_numpy() + g_hmm * tick,
        "HMM micro-price (argmax regime)": fte.mid.to_numpy() + g_hmm_hard * tick,
        "linear L1-L5 imbalance (10s OLS)": fte.mid.to_numpy() + lin * tick,
    }
    ev = evaluate_micro(fte, tick, preds)

    # Adverse selection by regime: aggressor-signed price impact of trades after 10 s (ticks).
    trades = pd.concat([read_csv_any(p, TRADE_COLS) for p in sorted(__import__("glob").glob(os.path.join(raw, sym, "trades_*.csv.gz")))
                        if complete_gzip(p)], ignore_index=True)
    trades = trades[(trades.txn_ms >= meta["start_ms"]) & (trades.txn_ms <= meta["end_ms"])]
    A, k, deltas, lam = fill_intensity(f[tr], trades[trades.txn_ms < meta["split_ms"]], tick)
    sigma = float(np.std(dmid1[tr]))
    midx = f.set_index("sec")
    tsec = (trades.txn_ms // 1000).astype(np.int64)
    side = np.where(trades.buyer_is_maker == 1, -1.0, 1.0)
    m10 = midx["mid"].reindex(tsec + 10).to_numpy()
    rg = midx["regime"].reindex(tsec - 1).to_numpy()  # regime known before the trade
    impact = side * (m10 - trades.price.to_numpy()) / tick
    adv = pd.DataFrame({"regime": rg, "side": side, "impact10": impact, "test": trades.txn_ms.to_numpy() >= meta["split_ms"]}).dropna()
    # Aggressor-signed impact mixes information with drift; averaging the buy-side and
    # sell-side means cancels a common linear drift and leaves the symmetric
    # information (adverse-selection) component.
    g = adv[adv.test].groupby(["regime", "side"])["impact10"].agg(["mean", "count"]).unstack("side")
    adv_tab = pd.DataFrame({
        "regime": g.index.astype(int),
        "impact_buys": g[("mean", 1.0)].to_numpy(), "impact_sells": g[("mean", -1.0)].to_numpy(),
        "adverse_selection_ticks": 0.5 * (g[("mean", 1.0)] + g[("mean", -1.0)]).to_numpy(),
        "n_trades": (g[("count", 1.0)] + g[("count", -1.0)]).to_numpy().astype(int),
    })

    reg_tab = pd.DataFrame({
        "regime": range(hmm.K),
        "share_test": [np.mean(reg[te] == r) for r in range(hmm.K)],
        "drift_1s_ticks": drift,
        "mean_imb1c": hmm.means[:, 0] * scale[0], "mean_imb_deep": hmm.means[:, 1] * scale[1],
        "mean_ofi1": hmm.means[:, 2] * scale[2], "mean_ofi_deep": hmm.means[:, 3] * scale[3],
        "mean_tflow5": hmm.means[:, 4] * scale[4],
        "persistence": np.diag(hmm.A),
    })

    os.makedirs(models_dir, exist_ok=True)
    write_cpp_model(os.path.join(models_dir, f"{sym}.model.txt"), tick, hmm, scale, plain, micro, drift, A, k, sigma)
    rdir = os.path.join(results_dir, sym)
    os.makedirs(rdir, exist_ok=True)
    ev.to_csv(os.path.join(rdir, "microprice_eval.csv"), index=False)
    reg_tab.to_csv(os.path.join(rdir, "regimes.csv"), index=False)
    adv_tab.to_csv(os.path.join(rdir, "adverse_selection.csv"), index=False)
    hmm.save(os.path.join(rdir, "hmm.json"))
    summary = {"meta": meta, "fill_A": A, "fill_k": k, "sigma_ticks": sigma, "mirror": mirror.tolist(),
               "gstar_plain": plain.table().tolist(), "gstar_hmm": micro.table().tolist(),
               "lambda_curve": {"delta": deltas.tolist(), "lambda": lam.tolist()}}
    with open(os.path.join(rdir, "research.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    pd.set_option("display.width", 200)
    print(reg_tab.round(3).to_string(index=False))
    print("adverse selection (test): 10s aggressor-signed impact by regime; AS = mean of buy/sell impacts (ticks)")
    print(adv_tab.round(3).to_string(index=False))
    piv = ev.pivot(index="model", columns="horizon_s", values="skill_vs_mid")
    print("out-of-sample skill vs mid (1 - MSE/MSE_mid) by horizon (s):")
    print(piv.round(4).to_string())
    print(f"fill intensity A={A:.3f}/s k={k:.3f}/tick, sigma={sigma:.3f} ticks/sqrt(s)")
    _plots(rdir, sym, ev, deltas, lam, A, k, f, tick)
    return ev, reg_tab


def _plots(rdir, sym, ev, deltas, lam, A, k, f, tick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    evh = ev[ev.horizon_s != "next mid change"].astype({"horizon_s": float})
    for name, g in evh.groupby("model"):
        if name != "mid":
            ax[0].plot(g.horizon_s, g.skill_vs_mid, marker="o", label=name)
    ax[0].set_xscale("log"); ax[0].set_xlabel("horizon (s)"); ax[0].set_ylabel("skill vs mid"); ax[0].legend(fontsize=7)
    ax[0].set_title(f"{sym}: micro-price forecast skill (test)")
    ax[1].semilogy(deltas, lam, "o", label="empirical")
    ax[1].semilogy(deltas, A * np.exp(-k * deltas), "-", label=f"A e^(-k d): A={A:.2f}, k={k:.2f}")
    ax[1].set_xlabel("distance from mid (ticks)"); ax[1].set_ylabel("rate per second"); ax[1].legend()
    ax[1].set_title("fill intensity")
    seg = f.iloc[-3600:]
    ax[2].plot(seg.sec - seg.sec.iloc[0], seg.mid, lw=0.6, color="k")
    for r, c in zip(range(3), ["tab:red", "tab:gray", "tab:green"]):
        m = seg.regime == r
        ax[2].scatter((seg.sec - seg.sec.iloc[0])[m], seg.mid[m], s=2, color=c, label=["sell pressure", "neutral", "buy pressure"][r])
    ax[2].legend(markerscale=4, fontsize=7); ax[2].set_title("last hour: mid coloured by filtered regime"); ax[2].set_xlabel("s")
    plt.tight_layout()
    plt.savefig(os.path.join(rdir, "research.png"), dpi=120)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="DOGEUSDT,ADAUSDT,XRPUSDT,TRXUSDT")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    for sym in a.symbols.split(","):
        run_symbol(sym, a.raw, a.out, a.results, a.models)


if __name__ == "__main__":
    main()
