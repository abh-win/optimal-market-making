#!/usr/bin/env python3
"""Records Binance USD-M futures top-of-book depth (10 levels, 100 ms snapshots)
and individual market trades from the public websocket, into hourly gzip CSV files.

  record_binance.py --symbols dogeusdt,adausdt,xrpusdt,trxusdt --out data/raw [--hours 12]

Files: <out>/<SYMBOL>/depth_<YYYYmmdd_HH>.csv.gz
         recv_ns,event_ms,txn_ms,update_id,bp1..bp10,bq1..bq10,ap1..ap10,aq1..aq10
       <out>/<SYMBOL>/trades_<YYYYmmdd_HH>.csv.gz
         recv_ns,event_ms,txn_ms,trade_id,price,qty,buyer_is_maker
Reconnects automatically (Binance drops connections every 24h, or on errors).
"""
import argparse
import asyncio
import gzip
import json
import os
import time

import websockets

LEVELS = 10


class HourlyWriter:
    def __init__(self, path_fmt, header):
        self.path_fmt, self.header = path_fmt, header
        self.hour, self.fh = None, None

    def write(self, line):
        hour = time.strftime("%Y%m%d_%H", time.gmtime())
        if hour != self.hour:
            if self.fh:
                self.fh.close()
            path = self.path_fmt.format(hour=hour)
            new = not os.path.exists(path)
            self.fh = gzip.open(path, "at", compresslevel=5)
            if new:
                self.fh.write(self.header + "\n")
            self.hour = hour
        self.fh.write(line + "\n")

    def close(self):
        if self.fh:
            self.fh.close()


async def run(symbols, out, deadline):
    writers = {}
    for s in symbols:
        d = os.path.join(out, s.upper())
        os.makedirs(d, exist_ok=True)
        cols = ",".join([f"bp{i}" for i in range(1, LEVELS + 1)] + [f"bq{i}" for i in range(1, LEVELS + 1)] +
                        [f"ap{i}" for i in range(1, LEVELS + 1)] + [f"aq{i}" for i in range(1, LEVELS + 1)])
        writers[s, "depth"] = HourlyWriter(os.path.join(d, "depth_{hour}.csv.gz"), "recv_ns,event_ms,txn_ms,update_id," + cols)
        writers[s, "trade"] = HourlyWriter(os.path.join(d, "trades_{hour}.csv.gz"),
                                           "recv_ns,event_ms,txn_ms,trade_id,price,qty,buyer_is_maker")
    streams = "/".join(f"{s}@depth20@100ms/{s}@trade" for s in symbols)
    url = f"wss://fstream.binance.com/stream?streams={streams}"
    n = 0
    while time.time() < deadline:
        try:
            async with websockets.connect(url, open_timeout=15, ping_interval=20, max_size=2 ** 22) as ws:
                print(time.strftime("%H:%M:%S"), "connected", flush=True)
                while time.time() < deadline:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    recv = time.time_ns()
                    stream, d = msg["stream"], msg["data"]
                    sym = stream.split("@")[0]
                    if "@depth" in stream:
                        b, a = d["b"][:LEVELS], d["a"][:LEVELS]
                        if len(b) < LEVELS or len(a) < LEVELS:
                            continue
                        row = [str(recv), str(d["E"]), str(d["T"]), str(d["u"])]
                        row += [x[0] for x in b] + [x[1] for x in b] + [x[0] for x in a] + [x[1] for x in a]
                        writers[sym, "depth"].write(",".join(row))
                    elif d.get("X", "MARKET") == "MARKET":  # skip liquidation/insurance-fund prints
                        writers[sym, "trade"].write(f"{recv},{d['E']},{d['T']},{d['t']},{d['p']},{d['q']},{int(d['m'])}")
                    n += 1
                    if n % 200000 == 0:
                        print(time.strftime("%H:%M:%S"), f"{n} messages", flush=True)
        except Exception as e:  # reconnect on any error
            print(time.strftime("%H:%M:%S"), "reconnect after error:", repr(e), flush=True)
            await asyncio.sleep(3)
    for w in writers.values():
        w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="dogeusdt,adausdt,xrpusdt,trxusdt")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--hours", type=float, default=12)
    a = ap.parse_args()
    asyncio.run(run(a.symbols.lower().split(","), a.out, time.time() + a.hours * 3600))


if __name__ == "__main__":
    main()
