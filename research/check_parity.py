#!/usr/bin/env python3
"""C++ online signals vs Python research features: they must agree.

  check_parity.py SYMBOL [--split train]
Runs cpp/build/mm_dump_signals and compares per-second features and the
HMM filtered posteriors with the Python implementation.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from fit_models import hmm_matrix  # noqa: E402
from mmr.hmm import GaussianHMM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(__file__), "..")
    out = f"/tmp/parity_{a.symbol}.csv"
    subprocess.run([os.path.join(root, "cpp/build/mm_dump_signals"), "--symbol", a.symbol, "--split", a.split,
                    "--data", os.path.join(root, "data/processed"), "--model",
                    os.path.join(root, f"models/{a.symbol}.model.txt"), "--out", out], check=True)
    c = pd.read_csv(out)
    f = pd.read_parquet(os.path.join(root, f"data/processed/{a.symbol}.seconds.parquet"))
    f = f[f.split == a.split]
    m = c.merge(f, on="sec", suffixes=("_cpp", "_py"))
    # The C++ stream starts at the split boundary: skip the first 10 s (OFI / trade-flow warm-up).
    m = m.iloc[10:]
    report = {}
    for col in ["imb1", "imb_deep", "ofi1", "ofi_deep", "tflow5"]:
        d = np.abs(m[f"{col}_cpp"] - m[f"{col}_py"])
        report[col] = {"max_abs_diff": float(d.max()), "share_exact_1e-9": float(np.mean(d < 1e-9))}
    # Posteriors: run the Python filter on Python features from the same start.
    hmm = GaussianHMM.from_dict(json.load(open(os.path.join(root, f"results/{a.symbol}/hmm.json"))))
    scale = np.array([float(x) for x in open(os.path.join(root, f"models/{a.symbol}.model.txt"))
                      .read().split("feat_scale")[1].split("\n")[0].split()])
    # Python filter over the same seconds the C++ engine saw, aligned by second; the
    # first 60 s are excluded (filter warm-up from the split boundary).
    pcols = [col for col in c.columns if col.startswith("p")]
    cc = c[np.isfinite(c[pcols]).all(axis=1)]
    ff = f[f.sec >= c.sec.iloc[0]].reset_index(drop=True)
    post_py = pd.DataFrame(hmm.filter(hmm_matrix(ff) / scale), columns=pcols)
    post_py["sec"] = ff.sec.to_numpy()
    mm = cc.merge(post_py, on="sec", suffixes=("_cpp", "_py"))
    mm = mm[mm.sec >= c.sec.iloc[0] + 60]
    a_ = mm[[p + "_cpp" for p in pcols]].to_numpy()
    b_ = mm[[p + "_py" for p in pcols]].to_numpy()
    report["posterior_seconds_compared"] = int(len(mm))
    report["posterior_argmax_agreement"] = float(np.mean(a_.argmax(1) == b_.argmax(1)))
    report["posterior_max_abs_diff"] = float(np.abs(a_ - b_).max())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
