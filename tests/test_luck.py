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
