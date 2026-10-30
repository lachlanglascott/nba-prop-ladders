"""Combo explorer: price a grid of P/R/A combos for one player and rank by correlation lift.

Lift = P(all legs) / product of leg probabilities. A book that prices same-game legs closer to
independent than they really are under-prices high-lift combos; that's the original edge.
Only combos with a fair price in a bettable range are shown.

    uv run python -m nba_ladders.combos "Precious Achiuwa" --opponent NOP
    uv run python -m nba_ladders.combos "Precious Achiuwa" --opponent NOP --as-of 2026-10-29
"""

import argparse
import itertools
from datetime import date

import numpy as np
import polars as pl

from nba_ladders.price import simulate_player

LINES = {
    "pts": [5, 8, 10, 12, 15, 18, 20, 25, 30, 35],
    "reb": [3, 4, 5, 6, 8, 10, 12, 15],
    "ast": [2, 3, 4, 5, 6, 8, 10, 12],
}
FAIR_RANGE = (1.8, 15.0)
MIN_LEG_P = 0.25  # every leg must be a realistic outcome on its own
SHAPES = {"p/r": ("pts", "reb"), "p/a": ("pts", "ast"), "r/a": ("reb", "ast"),
          "p/r/a": ("pts", "reb", "ast")}


def _relevant(values: np.ndarray, lines: list[int]) -> list[int]:
    """Lines between the player's ~10th and ~97th percentile (skip near-certain/impossible)."""
    return [x for x in lines if 0.03 <= (values >= x).mean() <= 0.9]


def explore(stats: dict, fair_range=FAIR_RANGE, min_leg_p: float = MIN_LEG_P) -> pl.DataFrame:
    """Every combo of the shapes in SHAPES with a fair price in range; sorted by lift."""
    hit = {s: {x: stats[s] >= x for x in _relevant(stats[s], LINES[s])} for s in LINES}
    rows = []
    for shape, combo_stats in SHAPES.items():
        for xs in itertools.product(*(hit[s] for s in combo_stats)):
            masks = [hit[s][x] for s, x in zip(combo_stats, xs, strict=True)]
            leg_p = [float(m.mean()) for m in masks]
            if min(leg_p) < min_leg_p:
                continue
            p = float(np.logical_and.reduce(masks).mean())
            if p <= 0:
                continue
            indep = float(np.prod(leg_p))
            rows.append({
                "shape": shape,
                "combo": "/".join(f"{x}{s[0]}" for s, x in zip(combo_stats, xs, strict=True)),
                "p": p, "fair": 1 / p, "fair_indep": 1 / indep, "lift": p / indep,
                "legs": ", ".join(f"{x}{s[0]} {lp:.0%}" for s, x, lp in zip(combo_stats, xs, leg_p, strict=True)),
            })
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    return df.filter(pl.col("fair").is_between(*fair_range)).sort("lift", descending=True)


def best_combos(stats: dict, per_shape: int = 3, shapes=None, **kw) -> pl.DataFrame:
    """Top combos by lift within each shape (so 2-leg combos aren't buried under 3-leg ones)."""
    df = explore(stats, **kw)
    if df.is_empty():
        return df
    if shapes:
        df = df.filter(pl.col("shape").is_in(shapes))
    return df.group_by("shape", maintain_order=True).head(per_shape)


def single_lines(stats: dict) -> pl.DataFrame:
    rows = [{"line": f"{x}{s[0]}", "p": float((stats[s] >= x).mean())}
            for s in LINES for x in _relevant(stats[s], LINES[s])]
    return pl.DataFrame(rows).with_columns((1 / pl.col("p")).alias("fair"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("player")
    ap.add_argument("--opponent")
    ap.add_argument("--starter-out", action="store_true")
    ap.add_argument("--as-of", type=date.fromisoformat)
    ap.add_argument("--top", type=int, default=4, help="combos per shape")
    ap.add_argument("--max-fair", type=float, default=FAIR_RANGE[1])
    args = ap.parse_args()
    out = simulate_player(args.player, starter_out=args.starter_out, opponent=args.opponent,
                          as_of=args.as_of, n=60_000)
    if out is None:
        raise SystemExit(f"not enough history for {args.player!r}")
    stats, _ = out
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=40, tbl_width_chars=160,
                   fmt_str_lengths=40):
        print("\nBest combos per shape by correlation lift "
              "(fair = model; fair_indep = what pricing the legs as independent gives):")
        print(best_combos(stats, per_shape=args.top, fair_range=(FAIR_RANGE[0], args.max_fair))
              .with_columns(pl.col("p", "fair", "fair_indep", "lift").round(2)))


if __name__ == "__main__":
    main()
