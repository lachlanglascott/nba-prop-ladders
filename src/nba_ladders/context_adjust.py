"""Line up a player's history with tonight's lineup before pricing.

A past game where a *current* teammate sat (Cade out -> Jenkins 34 min, 8 ast) shouldn't count
as a normal game when that teammate plays tonight. So: remove the effect of past absences of
players on tonight's roster (minutes shift and per-minute rate multipliers from absences.py),
then the caller adds tonight's absences back. Absences of teammates who have since left stay
in the history: they're gone tonight too, so that role is real (fixes the Diabaté/Achiuwa
under-projection the full context model had).
"""

import polars as pl

from nba_ladders.absences import RATES, context_effects, game_absences


def history_adjustment(pg: pl.DataFrame, hist: pl.DataFrame, roster_ids: set[int], eff, pri,
                       pos) -> pl.DataFrame:
    """Per history game: shift_min and mult_<stat> from absences of current-roster teammates."""
    absent = game_absences(pg).filter(pl.col("mate_id").is_in(list(roster_ids)))
    rows = hist.select("player_id", "game_id", "team_id").unique()
    return context_effects(rows, absent, eff, pri, pos).select("game_id", "shift_min",
                                                               *[f"mult_{s}" for s in RATES])


def adjust_history(hist: pl.DataFrame, adj: pl.DataFrame) -> pl.DataFrame:
    """Back each game out to "all current teammates available": minutes minus the shift, and
    counts scaled so per-minute rates are divided by the absence multipliers."""
    h = hist.join(adj, on="game_id", how="left").with_columns(
        pl.col("shift_min").fill_null(0.0), *[pl.col(f"mult_{s}").fill_null(1.0) for s in RATES])
    new_min = (pl.col("minutes") - pl.col("shift_min")).clip(lower_bound=1.0)
    scale = new_min / pl.col("minutes").clip(lower_bound=1.0)
    return h.with_columns(
        *[(pl.col(s) / pl.col(f"mult_{s}") * scale).alias(s) for s in RATES],
        new_min.alias("minutes"),
    ).drop("shift_min", *[f"mult_{s}" for s in RATES])
