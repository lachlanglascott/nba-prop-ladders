from datetime import date

import numpy as np
import polars as pl

from nba_ladders.market_backtest import flag, information_test, roi


def _df(n=4000, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.1, 0.6, n)
    p_book = np.clip(truth + rng.normal(0, 0.08, n), 0.02, 0.95)
    p_model = np.clip(truth + rng.normal(0, 0.03 if informative else 0.3, n), 0.02, 0.95)
    hit = rng.random(n) < truth
    return pl.DataFrame({"game_date": [date(2026, 1, 1 + i % 28) for i in range(n)],
                         "p_book": p_book, "p_model": p_model, "price": 1 / p_book * 0.92, "hit": hit})


def test_information_test_detects_model_signal():
    good = information_test(_df(informative=True))
    assert good["coef_model"] > 3 * good["coef_model_se"]
    assert good["logloss_book+model"] < good["logloss_book_only"]


def test_flag_respects_cushion_and_tail_guards():
    df = _df()
    bets = flag(df, (0.10, 0.15, 0.25))
    ev = bets["price"] * bets["p_model"] - 1
    assert (ev >= 0.10).all() and (ev < 1.0).all() and (1 / bets["p_model"] <= 15).all()


def test_roi_on_informed_model_is_positive_with_ci():
    r = roi(flag(_df(), (0.10, 0.15, 0.25)), n_boot=300)
    assert r["bets"] > 100 and r["roi"] > 0 and r["roi_ci90"][0] < r["roi"] < r["roi_ci90"][1]
