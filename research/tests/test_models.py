import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmr.hmm import GaussianHMM  # noqa: E402
from mmr.microprice import MicroPrice  # noqa: E402


def simulate_hmm(T=6000, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.95, 0.04, 0.01], [0.03, 0.94, 0.03], [0.01, 0.04, 0.95]])
    means = np.array([[-1.5, -1.0], [0.0, 0.0], [1.5, 1.0]])
    z = np.zeros(T, dtype=int)
    for t in range(1, T):
        z[t] = rng.choice(3, p=A[z[t - 1]])
    X = means[z] + rng.normal(0, 0.6, (T, 2))
    return X, z, A


def test_hmm_recovers_states_and_transitions():
    X, z, A = simulate_hmm()
    m = GaussianHMM(3, n_iter=200, n_init=2, seed=1).fit([X])
    m.reorder(np.argsort(m.means[:, 0]))
    acc_viterbi = np.mean(m.viterbi(X) == z)
    acc_filter = np.mean(m.filter(X).argmax(1) == z)
    assert acc_viterbi > 0.95
    assert acc_filter > 0.9            # causal filter is slightly worse than smoothing
    assert np.abs(m.A - A).max() < 0.05


def test_filter_is_causal():
    X, _, _ = simulate_hmm(2000)
    m = GaussianHMM(3, n_iter=50, n_init=1).fit([X])
    full = m.filter(X)
    prefix = m.filter(X[:1000])
    assert np.allclose(full[:1000], prefix)  # future data never changes past posteriors


def test_mirror_augmented_fit_is_symmetric():
    X, _, _ = simulate_hmm(4000)
    m = GaussianHMM(3, n_iter=100, n_init=2).fit([X, -X])
    m.reorder(np.argsort(m.means[:, 0]))
    assert np.allclose(m.means[0], -m.means[2], atol=0.1)
    assert np.allclose(m.means[1], 0, atol=0.1)


def make_book_transitions(n=60000, seed=0):
    """Toy book: imbalance predicts the next mid move (up with prob = imbalance)."""
    rng = np.random.default_rng(seed)
    imb = rng.uniform(0, 1, n)
    move = rng.uniform(0, 1, n) < 0.3
    up = rng.uniform(0, 1, n) < imb
    mid = np.cumsum(np.where(move, np.where(up, 1.0, -1.0), 0.0)) * 0.01
    return pd.DataFrame({"imb1": imb, "spread": 1, "mid": np.concatenate([[0], mid[:-1]]) + 100})


def test_microprice_antisymmetric_and_monotone():
    f = make_book_transitions()
    mp = MicroPrice(n_imb=5, n_spread=1)
    mp.fit(mp.transitions(f, np.zeros(len(f), dtype=int), tick=0.01))
    g = mp.table()[0, 0]
    assert np.allclose(g, -g[::-1], atol=1e-9)  # G*(1-I) = -G*(I) by symmetrisation
    assert np.all(np.diff(g) > 0)               # more bid depth -> higher fair price
    assert abs(g[2]) < 1e-9                     # balanced book -> no adjustment


def test_regime_mirroring_in_microprice():
    f = make_book_transitions()
    reg = np.random.default_rng(1).integers(0, 3, len(f))
    mp = MicroPrice(n_imb=5, n_spread=1, n_regimes=3, mirror=[2, 1, 0])
    mp.fit(mp.transitions(f, reg, tick=0.01))
    t = mp.table()
    assert np.allclose(t[0, 0], -t[2, 0, ::-1], atol=1e-9)  # sell regime mirrors buy regime


def test_symmetric_hmm_is_exactly_mirror_invariant():
    X, z, _ = simulate_hmm(4000)
    m = GaussianHMM(3, n_iter=100, n_init=2, symmetric=True).fit([X, -X])
    if m.means[0, 0] > m.means[2, 0]:
        m.reorder(np.array([2, 1, 0]))
    P = np.array([2, 1, 0])
    assert np.allclose(m.means[0], -m.means[2]) and np.allclose(m.means[1], 0)
    assert np.allclose(m.A, m.A[np.ix_(P, P)])
    assert np.mean(m.viterbi(X) == z) > 0.9
