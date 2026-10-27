"""Gaussian hidden Markov model (diagonal covariances), implemented from scratch.

Fitting: Baum-Welch (EM) with scaled forward-backward recursions, multiple
sequences, k-means initialisation and several restarts (best log-likelihood
wins). Inference: the *filtered* posterior P(z_t | x_1..t) is causal and is
what a live trading system (and our C++ engine) can use; Viterbi is provided
for analysis only.

For order-book regimes we fit on the data plus its mirror image (bid <-> ask,
which negates the signed features). The model then has (approximately)
mirror-symmetric states, e.g. "buy pressure" / "neutral" / "sell pressure",
which the symmetrised micro-price estimation needs.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans

LOG2PI = np.log(2 * np.pi)


class GaussianHMM:
    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-4, n_init: int = 3,
                 min_var: float = 1e-3, seed: int = 0, symmetric: bool = False):
        self.K, self.n_iter, self.tol, self.n_init, self.min_var, self.seed = n_states, n_iter, tol, n_init, min_var, seed
        # symmetric=True (K = 3): states are constrained to (-m, 0, +m) with tied
        # variances and a mirror-invariant transition matrix - the exact symmetry
        # of an order book under bid <-> ask exchange (fit on data + its mirror).
        self.symmetric = symmetric
        if symmetric and n_states != 3:
            raise ValueError("symmetric HMM requires 3 states (sell / neutral / buy)")

    # ------------------------------------------------------------------ core recursions
    def _log_emission(self, X):
        # log N(x | mu_k, diag var_k) for every t, k
        d = X[:, None, :] - self.means[None, :, :]
        return -0.5 * (np.sum(d * d / self.vars[None], axis=2) + np.sum(np.log(self.vars), axis=1)[None] +
                       X.shape[1] * LOG2PI)

    def _forward_backward(self, X):
        logB = self._log_emission(X)
        m = logB.max(axis=1, keepdims=True)
        B = np.exp(logB - m)  # rescaled emissions
        T = len(X)
        alpha = np.empty((T, self.K))
        c = np.empty(T)
        a = self.pi * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.empty((T, self.K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        loglik = np.sum(np.log(c)) + np.sum(m)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        # expected transitions
        xi = (alpha[:-1, :, None] * self.A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        return loglik, gamma, xi.sum(axis=0)

    def _em(self, seqs):
        prev = -np.inf
        for it in range(self.n_iter):
            ll, g_sum, x_sum, xx_sum, start, trans = 0.0, 0, 0, 0, 0, 0
            for X in seqs:
                l, gamma, xi = self._forward_backward(X)
                ll += l
                g_sum = g_sum + gamma.sum(0)
                x_sum = x_sum + gamma.T @ X
                xx_sum = xx_sum + gamma.T @ (X * X)
                start = start + gamma[0]
                trans = trans + xi
            self.pi = start / start.sum()
            self.A = trans / trans.sum(axis=1, keepdims=True)
            self.means = x_sum / g_sum[:, None]
            self.vars = np.maximum(xx_sum / g_sum[:, None] - self.means ** 2, self.min_var)
            if self.symmetric:
                self._symmetrize()
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        return ll, it + 1

    def _symmetrize(self):
        """Project onto the mirror-symmetric parameter space (P swaps states 0 and 2)."""
        P = np.array([2, 1, 0])
        m = 0.5 * (self.means[0] - self.means[2])
        self.means = np.stack([m, np.zeros_like(m), -m])
        v = 0.5 * (self.vars[0] + self.vars[2])
        self.vars = np.stack([v, self.vars[1], v])
        self.A = 0.5 * (self.A + self.A[np.ix_(P, P)])
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = 0.5 * (self.pi + self.pi[P])

    # ------------------------------------------------------------------ public API
    def fit(self, seqs: list[np.ndarray]) -> "GaussianHMM":
        allX = np.concatenate(seqs)
        best = None
        rng = np.random.default_rng(self.seed)
        for r in range(self.n_init):
            if self.symmetric:
                # (-m, 0, +m) along the first principal axis, m = mean of the upper tail.
                Xs = allX[::max(1, len(allX) // 50000)]
                u = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)[2][0]
                proj = Xs @ u
                q = np.quantile(proj, 0.75 + 0.05 * r)
                m = Xs[proj > q].mean(0) * (1 + 0.1 * rng.standard_normal())
                self.means = np.stack([-m, np.zeros_like(m), m])
                self.vars = np.tile(np.maximum(Xs.var(0), self.min_var), (3, 1))
            else:
                km = KMeans(self.K, n_init=4, random_state=int(rng.integers(1 << 30))).fit(allX[::max(1, len(allX) // 50000)])
                self.means = km.cluster_centers_.copy()
                self.vars = np.stack([np.maximum(allX[km.predict(allX) == k].var(0), self.min_var) for k in range(self.K)])
            self.pi = np.full(self.K, 1 / self.K)
            self.A = np.full((self.K, self.K), 0.05 / (self.K - 1)) + np.eye(self.K) * (0.95 - 0.05 / (self.K - 1))
            ll, iters = self._em(seqs)
            if best is None or ll > best[0]:
                best = (ll, iters, self.pi.copy(), self.A.copy(), self.means.copy(), self.vars.copy())
        self.loglik, self.iters, self.pi, self.A, self.means, self.vars = best
        return self

    def filter(self, X: np.ndarray) -> np.ndarray:
        """Causal posteriors P(z_t | x_1..t)."""
        logB = self._log_emission(X)
        B = np.exp(logB - logB.max(axis=1, keepdims=True))
        out = np.empty((len(X), self.K))
        a = self.pi * B[0]
        out[0] = a / a.sum()
        for t in range(1, len(X)):
            a = (out[t - 1] @ self.A) * B[t]
            out[t] = a / a.sum()
        return out

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        logB = self._log_emission(X)
        logA = np.log(self.A)
        T = len(X)
        delta = np.log(self.pi) + logB[0]
        back = np.empty((T, self.K), dtype=int)
        for t in range(1, T):
            s = delta[:, None] + logA
            back[t] = s.argmax(0)
            delta = s.max(0) + logB[t]
        path = np.empty(T, dtype=int)
        path[-1] = delta.argmax()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def reorder(self, order: np.ndarray) -> "GaussianHMM":
        """Permute state labels (e.g. sort by mean imbalance)."""
        self.pi, self.means, self.vars = self.pi[order], self.means[order], self.vars[order]
        self.A = self.A[np.ix_(order, order)]
        return self

    def to_dict(self) -> dict:
        return {"n_states": self.K, "pi": self.pi.tolist(), "A": self.A.tolist(), "means": self.means.tolist(),
                "vars": self.vars.tolist(), "loglik": float(self.loglik)}

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianHMM":
        m = cls(d["n_states"])
        m.pi, m.A = np.array(d["pi"]), np.array(d["A"])
        m.means, m.vars = np.array(d["means"]), np.array(d["vars"])
        m.loglik = d.get("loglik", np.nan)
        return m

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
