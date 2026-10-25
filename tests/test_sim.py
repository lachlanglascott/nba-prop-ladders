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


def test_parse_combo():
    from nba_ladders.sim import parse_combo

    assert parse_combo("20p/10r/3a") == {"pts": 20, "reb": 10, "ast": 3}
    assert parse_combo("15P / 8R") == {"pts": 15, "reb": 8}


def test_three_leg_joint_beats_independent():
    from nba_ladders.sim import combo_probs, simulate_stats

    h = _history().with_columns((pl.col("minutes") * 0.15).round().cast(pl.Int64).alias("ast"))
    s = simulate_stats(fit_player(h), starter_out=True, n=40000)
    row = combo_probs(s, [{"pts": 10, "reb": 8, "ast": 3}]).row(0, named=True)
    assert row["p"] > row["p_indep"] * 1.3


def test_recency_weights_halve_every_half_life():
    from nba_ladders.sim import recency_weights

    w = recency_weights(21, 10, floor=0.0)
    assert abs(w.sum() - 1) < 1e-12
    assert abs(w[-11] / w[-1] - 0.5) < 1e-12 and abs(w[0] / w[-1] - 0.25) < 1e-12
    assert recency_weights(5, None) is None
    # a floor keeps old games in play
    assert recency_weights(100, 5, floor=0.3)[0] >= 0.3 / 100


def test_recency_moves_simulation_toward_recent_form():
    # 40 old games at 10 min, 10 recent games at 30 min
    h = pl.DataFrame({"minutes": [10] * 40 + [30] * 10, "pts": [4] * 40 + [12] * 10,
                      "reb": [3] * 40 + [9] * 10, "starter_out": [False] * 50})
    flat, _, _ = simulate(fit_player(h, half_life=0), starter_out=False, n=20000)
    recent, _, _ = simulate(fit_player(h, half_life=5), starter_out=False, n=20000)
    assert recent.mean() > flat.mean() + 8


def test_points_can_be_odd_and_keep_scoring_variance():
    import numpy as np

    from nba_ladders.sim import draw_points

    pts = draw_points(np.random.default_rng(0), np.full(200_000, 15.0))
    assert abs(pts.mean() - 15) < 0.05
    assert (pts % 2 == 1).mean() > 0.4  # odd totals happen
    assert 1.9 < pts.var() / pts.mean() < 2.3
