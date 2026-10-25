"""As-at backtest: context-aware simulator vs the no-context baseline, all players.

    uv run python -m nba_ladders.backtest_context

Absence effects are refit at each monthly checkpoint from games before it. Who was absent in a
game comes from the box scores over the whole dataset (rostered but didn't play), standing in
for the pre-game injury report; using only data up to the checkpoint would miss absences of
players who hadn't returned yet and bias predictions down. Teammates' usual minutes use their
season average (mild look-ahead).
"""

from datetime import date
from itertools import pairwise

import polars as pl

from nba_ladders import config
from nba_ladders.absences import (
    RATES,
    context_effects,
    game_absences,
    pair_effects,
    priors,
    shrunk_effects,
)
from nba_ladders.context_sim import fit_context, simulate_context
from nba_ladders.scenarios import positions
from nba_ladders.sim import combo_probs, fit_player, parse_combo, simulate_stats

HISTORY_GAMES = 120
MIN_AVG_MINUTES = 12
COMBOS = ["10p", "15p", "20p", "8r", "15p/8r", "10p/10r", "15p/8r/2a", "20p/5r/5a"]


def checkpoints(pg: pl.DataFrame, seasons: list[str]) -> list[date]:
    days = pg.filter(pl.col("season_id").is_in(seasons))["game_date"]
    out, d = [], days.min().replace(day=1)
    while d <= days.max():
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out + [date(days.max().year + 1, 1, 1)]


def run(pg: pl.DataFrame, seasons=("22024", "22025"), n_sims: int = 3000,
        scale: float | None = None, half_life: float | None = None) -> pl.DataFrame:
    pos = positions()
    combos = [parse_combo(c) for c in COMBOS]
    rows = []
    cps = checkpoints(pg, list(seasons))
    absent = game_absences(pg)
    for start, end in pairwise(cps):
        before = pg.filter(pl.col("game_date") < start)
        upto = pg.filter(pl.col("game_date") < end)
        pri = priors(pair_effects(before, pos))
        eff = shrunk_effects(pair_effects(before, pos), pri)
        ctx = context_effects(upto, absent, eff, pri, pos, scale=scale)
        data = upto.join(ctx, on=["player_id", "game_id"]).sort("player_id", "game_date")
        for (pid,), g in data.group_by(["player_id"], maintain_order=True):
            dates = g["game_date"].to_list()
            for i, d in enumerate(dates):
                if not (start <= d < end) or g["season_id"][i] not in seasons:
                    continue
                hist = g.slice(max(0, i - HISTORY_GAMES), min(i, HISTORY_GAMES))
                if hist.height < 10 or hist["minutes"].mean() < MIN_AVG_MINUTES:
                    continue
                cf = fit_context(hist, half_life)
                bf = fit_player(hist.drop("starter_out", strict=False), half_life)
                if cf is None or bf is None:
                    continue
                t = g.row(i, named=True)
                mult = {s: t[f"mult_{s}"] for s in RATES}
                cs = simulate_context(cf, t["shift_min"], mult, n_sims, seed=i)
                bs = simulate_stats(bf, False, n_sims, seed=i)
                pc, pb = combo_probs(cs, combos), combo_probs(bs, combos)
                for j, c in enumerate(combos):
                    rows.append({
                        "player_id": pid, "game_id": t["game_id"], "game_date": d,
                        "combo": COMBOS[j], "shift": t["shift_min"], "n_absent": t["n_absent"],
                        "freed": t["freed_min"],
                        "min_actual": t["minutes"], "min_ctx": float(cs["minutes"].mean()),
                        "min_base": float(bs["minutes"].mean()),
                        "p_ctx": pc["p"][j], "p_base": pb["p"][j],
                        "hit": all(t[k] >= v for k, v in c.items()),
                    })
        print(f"{start}: {len(rows):,} rows", flush=True)
    return pl.DataFrame(rows)


def summary(res: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    eps = 1e-4
    ll = lambda c: -(pl.col("hit") * pl.col(c).clip(eps, 1 - eps).log()
                     + (~pl.col("hit")) * (1 - pl.col(c).clip(eps, 1 - eps)).log()).mean()
    return res.group_by(by).agg(
        pl.len().alias("n"), pl.col("hit").sum().alias("hits"),
        pl.col("p_ctx").sum().round(0).alias("exp_ctx"), pl.col("p_base").sum().round(0).alias("exp_base"),
        ll("p_ctx").round(4).alias("ll_ctx"), ll("p_base").round(4).alias("ll_base"),
    ).sort(by)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["22024", "22025"])
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--half-life", type=float, default=None)
    ap.add_argument("--out", default="backtest_context.parquet")
    args = ap.parse_args()
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    res = run(pg, tuple(args.seasons), scale=args.scale, half_life=args.half_life)
    out = config.DATA_PROCESSED / args.out
    res.write_parquet(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
