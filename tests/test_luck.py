from datetime import date, timedelta

import polars as pl
import pytest

from nba_ladders.luck import rolling_features


def _games(n=12):
    d0 = date(2025, 1, 1)
    return pl.DataFrame({
        "player_id": [1] * n,
        "game_date": [d0 + timedelta(days=i) for i in range(n)],
        "minutes": [30] * n,
        "ast": [2] * (n - 1) + [20],          # last game is a huge outlier
        "potential_ast": [8] * n,
        "reb": [5] * n,
        "reb_chances": [10] * n,
    })


def test_features_use_only_earlier_games():
    lg = {"ast": 0.5, "reb": 0.5}
    f = rolling_features(_games(), lg, window=10)
    last = f.row(-1, named=True)
    # the 20-assist game must not leak into its own prediction
    assert last["w_ast"] == 20 and last["w_potential_ast"] == 80 and last["w_min"] == 300
    assert last["pred_ast_actual"] == pytest.approx(2.0)


def test_unlucky_passer_gets_higher_expected_than_actual():
    lg = {"ast": 0.5, "reb": 0.5}
    last = rolling_features(_games(), lg, window=10).row(-1, named=True)
    # 2 of 8 potential converted (25%) vs league 50%: shrunk conversion sits between
    assert 0.25 < last["conv_ast"] < 0.5
    assert last["pred_ast_expected"] > last["pred_ast_actual"]
    assert last["luck_ast"] < 0


def test_luck_multiplier_raises_unlucky_and_shrinks_small_samples():
    from nba_ladders.luck import luck_multipliers

    lg = {"ast": 0.5, "reb": 0.5}
    hist = pl.DataFrame({"player_id": [1, 2], "ast": [500, 500], "potential_ast": [1000, 1000],
                         "reb": [500, 500], "reb_chances": [1000, 1000]})
    cur = pl.DataFrame({
        "player_id": [1, 2],
        "ast": [10, 1], "potential_ast": [40, 4],   # both converting 25% vs 50% long-run
        "reb": [20, 20], "reb_chances": [40, 40],   # neutral
    })
    m = {r["player_id"]: r for r in luck_multipliers(cur, hist, lg).iter_rows(named=True)}
    assert m[1]["luck_ast"] > 1.2                      # unlucky passer, decent sample
    assert 1.0 < m[2]["luck_ast"] < m[1]["luck_ast"]   # same luck, tiny sample: shrunk
    assert abs(m[1]["luck_reb"] - 1.0) < 0.02          # neutral rebounder


def test_effective_luck_only_scales_current_season_share():
    import numpy as np

    from nba_ladders.luck import effective_luck

    w = np.array([1.0, 1.0, 1.0, 1.0])
    cur = np.array([False, False, False, True])
    assert abs(effective_luck(1.2, w, cur) - 1.05) < 1e-12   # 25% of weight is this season
    assert effective_luck(1.2, None, [True] * 3) == 1.2
