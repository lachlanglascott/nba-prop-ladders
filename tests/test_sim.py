import numpy as np
import polars as pl

from nba_ladders.sim import fit_player, rung_probs, simulate


def _history(n=60, seed=1):
    """Synthetic backup: ~15 min normally; pts and reb rates move together game to game."""
    rng = np.random.default_rng(seed)
    form = rng.normal(0, 1, n)
    minutes = rng.integers(10, 22, n)
    pts = np.round(minutes * (0.45 + 0.12 * form).clip(0.05)).astype(int)
    reb = np.round(minutes * (0.35 + 0.10 * form).clip(0.05)).astype(int)
    return pl.DataFrame({"minutes": minutes, "pts": pts, "reb": reb, "starter_out": [False] * n})


def test_simulate_shapes_and_bounds():
    fit = fit_player(_history())
    m, p, r = simulate(fit, starter_out=False, n=5000)
    assert len(m) == len(p) == len(r) == 5000
    assert m.min() >= 0 and m.max() <= 48 and p.min() >= 0 and r.min() >= 0


def test_starter_out_lifts_minutes_without_out_history():
    fit = fit_player(_history())
    m_in, _, _ = simulate(fit, starter_out=False, n=20000)
    m_out, _, _ = simulate(fit, starter_out=True, n=20000)
    assert m_out.mean() > m_in.mean() * 1.2


def test_correlated_rates_make_joint_beat_independent():
    fit = fit_player(_history())
    _, p, r = simulate(fit, starter_out=True, n=40000)
    probs = rung_probs(p, r, rungs=[(12, 10)])["12/10"]
    assert probs["joint"] > probs["indep"] * 1.2


def test_too_little_history_returns_none():
    assert fit_player(_history(n=3)) is None
