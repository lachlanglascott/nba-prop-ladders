"""As-at backtest of the simulator on backup-big games.

    uv run python -m nba_ladders.backtest
"""

import polars as pl

from nba_ladders import config
from nba_ladders.sim import RUNGS, fit_player, rung_probs, simulate

HISTORY_GAMES = 120
TEST_SEASONS = ["22024", "22025"]


def run(bg: pl.DataFrame, test_seasons=TEST_SEASONS, n_sims: int = 4000, **sim_opts) -> pl.DataFrame:
    """One row per (test game, rung): predicted joint/independent prob and the outcome.

    sim_opts are passed to simulate() (e.g. paired=False for the ablation).
    """
    bg = bg.sort("player_id", "game_date")
    rows = []
    for (player_id,), games in bg.group_by(["player_id"], maintain_order=True):
        for i, g in enumerate(games.iter_rows(named=True)):
            if g["season_id"] not in test_seasons:
                continue
            hist = games.slice(max(0, i - HISTORY_GAMES), min(i, HISTORY_GAMES))
            fit = fit_player(hist)
            if fit is None:
                continue
            _, pts, reb = simulate(fit, g["starter_out"], n_sims, seed=i, **sim_opts)
            for (a, b), (rung, p) in zip(RUNGS, rung_probs(pts, reb).items(), strict=True):
                rows.append({
                    "player_id": player_id, "player_name": g["player_name"],
                    "game_id": g["game_id"], "game_date": g["game_date"],
                    "season_id": g["season_id"], "starter_out": g["starter_out"],
                    "rung": rung, "p_joint": p["joint"], "p_indep": p["indep"],
                    "hit": (g["pts"] >= a) and (g["reb"] >= b),
                })
    return pl.DataFrame(rows)


def calibration(res: pl.DataFrame, prob: str, by: list[str] | None = None) -> pl.DataFrame:
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
    return (
        res.with_columns(pl.col(prob).cut(bins[1:-1]).alias("bucket"))
        .group_by([*(by or []), "bucket"])
        .agg(pl.len().alias("n"), pl.col(prob).mean().alias("predicted"),
             pl.col("hit").mean().alias("actual"))
        .sort([*(by or []), "predicted"])
    )


def scores(res: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Brier score and log loss for joint vs independent, plus totals predicted vs actual."""
    eps = 1e-4
    ll = lambda c: -(pl.col("hit") * pl.col(c).clip(eps, 1 - eps).log()
                     + (~pl.col("hit")) * (1 - pl.col(c).clip(eps, 1 - eps)).log()).mean()
    return res.group_by(by).agg(
        pl.len().alias("n"),
        pl.col("hit").sum().alias("hits"),
        pl.col("p_joint").sum().alias("exp_joint"),
        pl.col("p_indep").sum().alias("exp_indep"),
        ((pl.col("p_joint") - pl.col("hit")) ** 2).mean().alias("brier_joint"),
        ((pl.col("p_indep") - pl.col("hit")) ** 2).mean().alias("brier_indep"),
        ll("p_joint").alias("logloss_joint"),
        ll("p_indep").alias("logloss_indep"),
    ).sort(by)


def main() -> None:
    bg = pl.read_parquet(config.DATA_PROCESSED / "backup_big_games.parquet")
    res = run(bg)
    out = config.DATA_PROCESSED / "backtest_sim.parquet"
    res.write_parquet(out)
    print(f"{res['game_id'].n_unique():,} games, {res.height:,} rung predictions -> {out}")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, tbl_hide_column_data_types=True):
        print(scores(res, ["starter_out", "rung"]).with_columns(pl.col(pl.Float64).round(4)))
        print(calibration(res, "p_joint").with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
