"""Recorded Binance depth/trade CSVs -> compact binaries for the C++ backtester
and a 1-second feature table for the research models.

Binary layouts (little-endian float64, row-major):
  <SYM>.depth.bin   rows of 41: ts_ms, bp1..bp10, bq1..bq10, ap1..ap10, aq1..aq10
  <SYM>.trades.bin  rows of 4:  ts_ms, price, qty, side (+1 buyer-initiated, -1 seller-initiated)
Exchange transaction time (T) is used as the clock for both streams.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

LEVELS = 10


DEPTH_COLS = (["recv_ns", "event_ms", "txn_ms", "update_id"] + [f"bp{i}" for i in range(1, LEVELS + 1)] +
              [f"bq{i}" for i in range(1, LEVELS + 1)] + [f"ap{i}" for i in range(1, LEVELS + 1)] +
              [f"aq{i}" for i in range(1, LEVELS + 1)])
TRADE_COLS = ["recv_ns", "event_ms", "txn_ms", "trade_id", "price", "qty", "buyer_is_maker"]


def read_csv_any(path: str, cols: list[str]) -> pd.DataFrame:
    """Reads a recorder file with or without a header line (a file can lack the
    header if the recorder was restarted within the hour after an empty first member)."""
    import gzip
    with gzip.open(path, "rt") as fh:
        first = fh.readline()
    has_header = not first[:1].isdigit()
    df = pd.read_csv(path, header=0 if has_header else None, names=None if has_header else cols)
    # drop any repeated header rows from appended members
    return df[pd.to_numeric(df[cols[0]], errors="coerce").notna()].astype({cols[2]: "int64"}) if not has_header else df


def complete_gzip(path: str) -> bool:
    """True if the gzip stream is complete (the recorder closes files hourly;
    the file of the current hour is still open and unreadable to the end)."""
    import gzip
    if os.path.getsize(path) == 0:
        return False
    try:
        n = 0
        with gzip.open(path, "rb") as fh:
            while chunk := fh.read(1 << 20):
                n += len(chunk)
        return n > 0
    except (EOFError, OSError):
        return False


def load_raw(raw_dir: str, symbol: str, start: str | None = None, end: str | None = None):
    d = os.path.join(raw_dir, symbol)
    dfiles = [f for f in sorted(glob.glob(os.path.join(d, "depth_*.csv.gz"))) if complete_gzip(f)]
    tfiles = [f for f in sorted(glob.glob(os.path.join(d, "trades_*.csv.gz"))) if complete_gzip(f)]
    if not dfiles or not tfiles:
        raise RuntimeError(f"no complete recordings for {symbol} yet")
    depth = pd.concat([read_csv_any(f, DEPTH_COLS) for f in dfiles], ignore_index=True)
    trades = pd.concat([read_csv_any(f, TRADE_COLS) for f in tfiles], ignore_index=True)
    depth = depth.sort_values("txn_ms").drop_duplicates("update_id")
    trades = trades.sort_values(["txn_ms", "trade_id"]).drop_duplicates("trade_id")
    # Trades start a little after depth on the first file (stream switch); align.
    t0 = max(depth.txn_ms.iloc[0], trades.txn_ms.iloc[0])
    depth, trades = depth[depth.txn_ms >= t0], trades[trades.txn_ms >= t0]
    if start:
        ms = pd.Timestamp(start, tz="UTC").value // 1_000_000
        depth, trades = depth[depth.txn_ms >= ms], trades[trades.txn_ms >= ms]
    if end:
        ms = pd.Timestamp(end, tz="UTC").value // 1_000_000
        depth, trades = depth[depth.txn_ms < ms], trades[trades.txn_ms < ms]
    return depth.reset_index(drop=True), trades.reset_index(drop=True)


def infer_tick(depth: pd.DataFrame) -> float:
    px = np.unique(np.concatenate([depth[f"bp{i}"].to_numpy()[:20000] for i in range(1, 6)]))
    d = np.diff(px)
    d = d[d > 0]
    # round to the decimal grid (prices are printed with fixed decimals)
    return float(np.round(np.min(d), 10))


def write_binaries(depth: pd.DataFrame, trades: pd.DataFrame, out_prefix: str) -> None:
    cols = ["txn_ms"] + [f"bp{i}" for i in range(1, LEVELS + 1)] + [f"bq{i}" for i in range(1, LEVELS + 1)] + \
           [f"ap{i}" for i in range(1, LEVELS + 1)] + [f"aq{i}" for i in range(1, LEVELS + 1)]
    depth[cols].to_numpy(np.float64).tofile(out_prefix + ".depth.bin")
    side = np.where(trades["buyer_is_maker"].to_numpy() == 1, -1.0, 1.0)
    np.column_stack([trades["txn_ms"].to_numpy(np.float64), trades["price"].to_numpy(np.float64),
                     trades["qty"].to_numpy(np.float64), side]).tofile(out_prefix + ".trades.bin")


def second_features(depth: pd.DataFrame, trades: pd.DataFrame, tick: float) -> pd.DataFrame:
    """One row per second (last snapshot in the second) with the model inputs.

    imb1       L1 imbalance bq1/(bq1+aq1) in [0, 1] (Stoikov's I)
    imb_deep   depth imbalance of levels 2-5: (sum bq2..5 - sum aq2..5) / total, in [-1, 1]
    imb_k      cumulative imbalance of levels 1..k, k = 1..5, in [-1, 1]
    ofi1/ofi_deep  Cont-Kukanov-Stoikov order-flow imbalance over the second at level 1 and
               summed over levels 2-5 (multi-level OFI), normalised by mean L1-5 depth
    tflow5     signed trade volume share over the last 5 seconds
    """
    sec = (depth["txn_ms"].to_numpy() // 1000).astype(np.int64)
    bp = depth[[f"bp{i}" for i in range(1, 6)]].to_numpy()
    bq = depth[[f"bq{i}" for i in range(1, 6)]].to_numpy()
    ap = depth[[f"ap{i}" for i in range(1, 6)]].to_numpy()
    aq = depth[[f"aq{i}" for i in range(1, 6)]].to_numpy()

    # Level-wise OFI between consecutive snapshots.
    def lvl_ofi(k):
        b0, b1, q0b, q1b = bp[:-1, k], bp[1:, k], bq[:-1, k], bq[1:, k]
        a0, a1, q0a, q1a = ap[:-1, k], ap[1:, k], aq[:-1, k], aq[1:, k]
        e = (b1 >= b0) * q1b - (b1 <= b0) * q0b - (a1 <= a0) * q1a + (a1 >= a0) * q0a
        return np.concatenate([[0.0], e])
    ofi = np.stack([lvl_ofi(k) for k in range(5)], axis=1)
    depth_tot = bq.sum(1) + aq.sum(1)

    snap = pd.DataFrame({
        "sec": sec,
        "bid": bp[:, 0], "ask": ap[:, 0], "bq1": bq[:, 0], "aq1": aq[:, 0],
        "ofi1": ofi[:, 0], "ofi_deep": ofi[:, 1:].sum(1), "depth": depth_tot,
    })
    for k in range(1, 6):
        snap[f"imb_{k}"] = (bq[:, :k].sum(1) - aq[:, :k].sum(1)) / (bq[:, :k].sum(1) + aq[:, :k].sum(1))
    snap["imb_deep"] = (bq[:, 1:].sum(1) - aq[:, 1:].sum(1)) / (bq[:, 1:].sum(1) + aq[:, 1:].sum(1))

    g = snap.groupby("sec")
    f = g.last()
    f["ofi1"] = g["ofi1"].sum() / g["depth"].mean() * 10
    f["ofi_deep"] = g["ofi_deep"].sum() / g["depth"].mean() * 10
    f = f.reindex(np.arange(f.index.min(), f.index.max() + 1))
    f[["ofi1", "ofi_deep"]] = f[["ofi1", "ofi_deep"]].fillna(0.0)
    f = f.ffill()

    tsec = (trades["txn_ms"].to_numpy() // 1000).astype(np.int64)
    signed = np.where(trades["buyer_is_maker"].to_numpy() == 1, -1.0, 1.0) * trades["qty"].to_numpy()
    tv = pd.DataFrame({"sec": tsec, "signed": signed, "vol": trades["qty"].to_numpy()}).groupby("sec").sum()
    tv = tv.reindex(f.index).fillna(0.0)
    roll = tv.rolling(5, min_periods=1).sum()
    f["tflow5"] = (roll["signed"] / roll["vol"].replace(0, np.nan)).fillna(0.0)
    f["vol1s"] = tv["vol"]

    f["mid"] = (f["bid"] + f["ask"]) / 2
    f["spread"] = np.round((f["ask"] - f["bid"]) / tick).astype(int)
    f["imb1"] = f["bq1"] / (f["bq1"] + f["aq1"])
    f["wmid"] = f["imb1"] * f["ask"] + (1 - f["imb1"]) * f["bid"]
    f.index.name = "sec"
    return f.reset_index().drop(columns=["depth"])


def prepare(raw_dir: str, symbol: str, out_dir: str, split_frac: float = 0.65) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    depth, trades = load_raw(raw_dir, symbol)
    tick = infer_tick(depth)
    t0, t1 = depth.txn_ms.iloc[0], depth.txn_ms.iloc[-1]
    cut = int(t0 + split_frac * (t1 - t0))
    meta = {"symbol": symbol, "tick": tick, "start_ms": int(t0), "end_ms": int(t1), "split_ms": cut,
            "n_depth": int(len(depth)), "n_trades": int(len(trades)),
            "hours": round((t1 - t0) / 3.6e6, 2)}
    for name, sel_d, sel_t in [("train", depth.txn_ms < cut, trades.txn_ms < cut),
                               ("test", depth.txn_ms >= cut, trades.txn_ms >= cut)]:
        write_binaries(depth[sel_d], trades[sel_t], os.path.join(out_dir, f"{symbol}.{name}"))
    feats = second_features(depth, trades, tick)
    feats["split"] = np.where(feats["sec"] * 1000 < cut, "train", "test")
    feats.to_parquet(os.path.join(out_dir, f"{symbol}.seconds.parquet"), index=False)
    with open(os.path.join(out_dir, f"{symbol}.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta
