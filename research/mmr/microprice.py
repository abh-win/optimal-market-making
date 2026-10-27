"""Stoikov (2018) micro-price, extended with a latent order-flow regime.

Base model (Stoikov 2018, "The micro-price"): the state x = (imbalance bucket,
spread in ticks). From one-step transitions estimate
    Q[x,y] = P(mid unchanged, next state y | x)
    R[x,k] = P(mid jumps by K_k | x)
    T[x,y] = P(mid changed, next state y | x)
then G1 = (I-Q)^-1 R K, B = (I-Q)^-1 T and G* = sum_{i>=0} B^i G1; the
micro-price is mid + G*(x). Training data is symmetrised (I -> 1-I, dM -> -dM)
so that B's spectral radius is < 1 and the series converges.

Rare states (< min_count observations) are made absorbing with zero adjustment,
and regime-specific G* values are shrunk towards the pooled (plain Stoikov) value
of the same cell with weight n / (n + shrink), so sparsely observed regime cells
fall back to the well-estimated model.

Extension: the state is augmented with the HMM regime z (from causal filtered
posteriors), x = (imbalance bucket, spread, z). Mirror symmetry maps regime z
to its mirror image mirror[z] (buy pressure <-> sell pressure), so the
symmetrised estimation stays consistent. The regime carries information from
deep-book imbalances (levels 2-5), multi-level OFI and trade flow that the
level-1 imbalance alone does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MicroPrice:
    def __init__(self, n_imb: int = 5, n_spread: int = 2, n_regimes: int = 1, mirror=None, max_dm_ticks: float = 1.0,
                 min_count: int = 50, shrink: float = 200.0):
        self.n_imb, self.n_spread, self.n_reg = n_imb, n_spread, n_regimes
        self.mirror = np.arange(n_regimes) if mirror is None else np.asarray(mirror)
        self.edges = np.linspace(0, 1, n_imb + 1)[1:-1]
        self.max_dm = max_dm_ticks
        self.min_count, self.shrink = min_count, shrink

    @property
    def n_states(self):
        return self.n_imb * self.n_spread * self.n_reg

    def _bucket(self, imb):
        return np.searchsorted(self.edges, imb, side="right")

    def state(self, imb, spread, reg):
        b = self._bucket(np.asarray(imb))
        s = np.clip(np.asarray(spread), 1, self.n_spread) - 1
        return (np.asarray(reg) * self.n_spread + s) * self.n_imb + b

    def transitions(self, f: pd.DataFrame, reg: np.ndarray, tick: float, step: int = 1) -> pd.DataFrame:
        """One-step transitions on the 1-second grid, |dM| <= max_dm, symmetrised."""
        imb, spr, mid = f["imb1"].to_numpy(), f["spread"].to_numpy(), f["mid"].to_numpy()
        dm = np.round((mid[step:] - mid[:-step]) / tick * 2) / 2  # ticks, half-tick resolution
        t = pd.DataFrame({"imb": imb[:-step], "spread": spr[:-step], "reg": reg[:-step],
                          "imb2": imb[step:], "spread2": spr[step:], "reg2": reg[step:], "dm": dm})
        t = t[(t.spread <= self.n_spread) & (t.spread2 <= self.n_spread) & (np.abs(t.dm) <= self.max_dm)]
        flip = t.assign(imb=1 - t.imb, imb2=1 - t.imb2, reg=self.mirror[t.reg], reg2=self.mirror[t.reg2], dm=-t.dm)
        return pd.concat([t, flip], ignore_index=True)

    def fit(self, t: pd.DataFrame) -> "MicroPrice":
        n = self.n_states
        x = self.state(t.imb, t.spread, t.reg)
        y = self.state(t.imb2, t.spread2, t.reg2)
        dm = t.dm.to_numpy()
        self.K = np.array(sorted(set(np.round(dm[dm != 0], 1))))
        move = dm != 0
        Qc, Tc = np.zeros((n, n)), np.zeros((n, n))
        Rc = np.zeros((n, len(self.K)))
        np.add.at(Qc, (x[~move], y[~move]), 1)
        np.add.at(Tc, (x[move], y[move]), 1)
        np.add.at(Rc, (x[move], np.searchsorted(self.K, np.round(dm[move], 1))), 1)
        self.counts = Qc.sum(1) + Tc.sum(1)
        den = np.maximum(self.counts, 1)[:, None]
        Q, T, R = Qc / den, Tc / den, Rc / den
        # Rarely visited states: absorbing with zero adjustment (their rows are too
        # noisy to invert; with regimes, some (imbalance, spread, regime) cells are rare).
        empty = self.counts < self.min_count
        Q[empty] = 0
        R[empty] = 0
        T[empty] = 0
        eye = np.eye(n)
        G1 = np.linalg.solve(eye - Q, R @ self.K)
        B = np.linalg.solve(eye - Q, T)
        g, term = G1.copy(), G1.copy()
        for _ in range(5000):
            term = B @ term
            g += term
            if np.abs(term).max() < 1e-12:
                break
        self.G1, self.gstar = G1, g
        if self.n_reg > 1:
            # Shrink each regime's adjustment towards the regime-pooled (plain Stoikov)
            # G* in the same (imbalance, spread) cell: w = n / (n + shrink).
            pooled = MicroPrice(self.n_imb, self.n_spread, 1, max_dm_ticks=self.max_dm, min_count=self.min_count)
            pooled.fit(t.assign(reg=0, reg2=0))
            base = np.tile(pooled.gstar, self.n_reg)
            w = self.counts / (self.counts + self.shrink)
            self.gstar = w * self.gstar + (1 - w) * base
        return self

    def adjust(self, imb, spread, reg) -> np.ndarray:
        """G* in ticks (NaN outside the modelled spread range)."""
        spread = np.asarray(spread)
        out = self.gstar[self.state(imb, np.clip(spread, 1, self.n_spread), reg)].astype(float)
        return np.where((spread >= 1) & (spread <= self.n_spread), out, np.nan)

    def table(self) -> np.ndarray:
        """gstar as [regime, spread, imb_bucket] for export."""
        return self.gstar.reshape(self.n_reg, self.n_spread, self.n_imb)
