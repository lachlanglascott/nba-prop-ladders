import numpy as np

from nba_ladders.combos import best_combos, explore


def _stats(n=50_000, seed=0):
    rng = np.random.default_rng(seed)
    minutes = rng.normal(27, 6, n).clip(5, 44)
    return {"minutes": minutes, "pts": rng.poisson(0.5 * minutes), "reb": rng.poisson(0.3 * minutes),
            "ast": rng.poisson(0.08 * minutes)}


def test_lift_above_one_when_legs_share_minutes():
    df = explore(_stats())
    pr = df.filter(df["shape"] == "p/r")
    assert pr.height > 0 and pr["lift"].min() > 1.0  # minutes drive both legs


def test_fair_range_and_realistic_legs():
    df = explore(_stats(), fair_range=(2.0, 10.0), min_leg_p=0.3)
    assert df["fair"].min() >= 2.0 and df["fair"].max() <= 10.0
    for legs in df["legs"]:
        assert all(int(part.split()[-1].rstrip("%")) >= 30 for part in legs.split(", "))


def test_best_combos_keeps_each_shape():
    b = best_combos(_stats(), per_shape=2)
    assert set(b["shape"]) >= {"p/r", "p/r/a"}
    assert b.group_by("shape").len()["len"].max() <= 2
