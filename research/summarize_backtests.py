#!/usr/bin/env python3
"""Summarise backtest CSVs into README tables and frontier plots.

  summarize_backtests.py [--symbols ...] [--results results]
Outputs results/summary_tables.md and results/<SYM>/frontier.png
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="DOGEUSDT,ADAUSDT,XRPUSDT,TRXUSDT")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    for sym in a.symbols.split(","):
        rdir = os.path.join(a.results, sym)
        meta = json.load(open(os.path.join("data/processed", f"{sym}.meta.json")))
        out.append(f"### {sym}\n")
        for keep in (0, 2):
            path = os.path.join(rdir, f"backtest_keep{keep}.csv")
            if not os.path.exists(path):
                continue
            b = pd.read_csv(path)
            label = "naive cancel/replace" if keep == 0 else "queue-priority retention (keep ≤ 2 ticks)"
            out.append(f"**{label}** (test split)\n")
            out.append("| strategy | params | PnL ($) | fills | PnL per fill (ticks) | spread capture ($) | inventory PnL ($) | markout 1s / 10s / 60s (ticks) |")
            out.append("|---|---|---|---|---|---|---|---|")
            for _, r in b.iterrows():
                per = r.pnl / max(r.fills, 1) / (r.lot * r.tick)  # $ per fill -> ticks per lot
                params = "" if pd.isna(r.params) else r.params
                out.append(f"| {r.strategy} | {params} | {r.pnl:+.2f} | {int(r.fills)} | "
                           f"{per:+.3f} | {r.spread_capture:.2f} | {r.inventory_pnl:+.2f} | "
                           f"{r.markout_1s:+.2f} / {r.markout_10s:+.2f} / {r.markout_60s:+.2f} |")
            out.append("")
        # frontier plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, keep in zip(axes, (0, 2)):
            fp = os.path.join(rdir, f"frontier_keep{keep}.csv")
            if not os.path.exists(fp):
                continue
            f = pd.read_csv(fp)
            for (fam, guard), g in f.groupby(["family", "guard"]):
                name = fam if guard < 0 else f"{fam} guard={int(guard)}"
                g = g.sort_values("test_fills")
                ax.plot(g.test_fills, g.test_pnl, marker="o", ms=3, label=name)
            bk = pd.read_csv(os.path.join(rdir, f"backtest_keep{keep}.csv"))
            fsr = bk[bk.strategy.str.startswith("fixed")]
            ax.scatter(fsr.fills, fsr.pnl, marker="x", s=80, color="k", label="fixed spread (join best)")
            ax.axhline(0, color="gray", lw=0.6)
            ax.set_xscale("symlog")
            ax.set_xlabel("test fills")
            ax.set_ylabel("test PnL ($, $100 lots)")
            ax.set_title(f"{sym} keep-ticks={keep}: gamma frontier")
            ax.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(rdir, "frontier.png"), dpi=120)
        plt.close("all")
    with open(os.path.join(a.results, "summary_tables.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
