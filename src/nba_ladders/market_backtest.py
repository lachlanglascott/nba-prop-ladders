"""Backtest model vs market on historical Sportsbet prices (The Odds API backfill).

    uv run python -m nba_ladders.market_backtest build      # price every historical rung as-at
    uv run python -m nba_ladders.market_backtest report     # accuracy, value-bet ROI, by angle

build: for each Sportsbet "X+" rung ~2h before tip, the model's probability using only data
before that game day (role shift, recency, as-at luck from per-game tracking, opponent
rebounds; not the absence-context model), the actual result, the US no-vig reference, and
angle tags. Players who didn't play are dropped (books void those bets).

report: (1) does the model add information beyond Sportsbet's price? logistic regression of
outcomes on both; (2) ROI of flagged VALUE bets at Sportsbet's price, cushions tuned on one
random half of game days and tested on the other, with bootstrap CIs, split by angle.
"""

import argparse
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from nba_ladders import config
from nba_ladders.luck import live_luck
from nba_ladders.market import CHECK_NEWS_EV, MAX_FAIR, TEAM_ABBR, model_p, us_reference
from nba_ladders.price import _frame, simulate_player
from nba_ladders.sources.injury_report import player_key
from nba_ladders.sources.odds_api import AU_BASE, load_cached

OUT = config.DATA_PROCESSED / "market_backtest.parquet"
ET = ZoneInfo("America/New_York")
LUCK_TAG = 1.05  # season-to-date luck multiplier at/above this tags the rung as a luck play


def _game_day(commence: str):
    return datetime.fromisoformat(commence).astimezone(ET).date()


def build(n_sims: int = 30_000) -> pl.DataFrame:
    odds = load_cached("historical")
    sb = odds.filter(pl.col("bookmaker") == AU_BASE, pl.col("side") == "Over",
                     pl.col("threshold").is_not_null())
    sb = sb.with_columns(pl.col("commence_time").map_elements(_game_day, return_dtype=pl.Date)
                         .alias("game_date"))
    ref = us_reference(odds).rename({"player": "player_api"})
    pg = _frame("player_games")
    bg = _frame("backup_big_games").select("player_id", "game_id", "starter_out")
    games = pg.with_columns(pl.col("player_name").map_elements(player_key, return_dtype=pl.String)
                            .alias("key"))
    rows = []
    days = sorted(sb["game_date"].unique().to_list())
    no_luck = pl.DataFrame(schema={"player_id": pl.Int64, "luck_ast": pl.Float64, "luck_reb": pl.Float64})
    for i, d in enumerate(days, 1):
        try:
            luck = live_luck(as_of=d)
        except FileNotFoundError:  # opening nights: no season-to-date tracking yet
            luck = no_luck
        day_games = games.filter(pl.col("game_date") == d)
        for (who, home, away), g in sb.filter(pl.col("game_date") == d).group_by(
                ["player", "home_team", "away_team"]):
            m = day_games.filter(pl.col("key") == player_key(who))
            if m.is_empty():
                continue  # didn't play (void) or name not matched
            r0 = m.row(0, named=True)
            h, a = TEAM_ABBR.get(home), TEAM_ABBR.get(away)
            opp = a if r0["team"] == h else h
            sim = simulate_player(r0["player_name"], opponent=opp, as_of=d, verbose=False, n=n_sims)
            if sim is None:
                continue
            stats, info = sim
            lk = luck.filter(pl.col("player_id") == r0["player_id"])
            la, lr = (lk["luck_ast"][0], lk["luck_reb"][0]) if lk.height else (1.0, 1.0)
            so = bg.filter((pl.col("player_id") == r0["player_id"]) & (pl.col("game_id") == r0["game_id"]))
            season_games = pg.filter((pl.col("player_id") == r0["player_id"]) & (pl.col("game_date") < d)
                                     & (pl.col("season_id") == r0["season_id"])).height
            for r in g.unique(["stat", "threshold"]).iter_rows(named=True):
                p = model_p(stats, r["stat"], r["threshold"])
                actual = sum(r0[s] for s in r["stat"].split("+"))
                rows.append({
                    "game_date": d, "event_id": r["event_id"], "player": r0["player_name"],
                    "player_api": who, "player_id": r0["player_id"], "team": r0["team"], "opp": opp,
                    "stat": r["stat"], "threshold": r["threshold"], "price": r["price"],
                    "p_model": p, "hit": actual >= r["threshold"],
                    "starter_out": bool(so["starter_out"][0]) if so.height else False,
                    "luck_ast": la, "luck_reb": lr, "season_games_before": season_games,
                    "history_games": info["games"],
                })
        print(f"{i}/{len(days)} {d}: {len(rows):,} rungs", flush=True)
    out = (pl.DataFrame(rows)
           .join(ref, on=["player_api", "stat", "threshold"], how="left")
           .with_columns(
               (1 / pl.col("price")).alias("p_book"),
               pl.when(pl.col("starter_out")).then(pl.lit("starter_out"))
               .when(((pl.col("stat").str.contains("reb")) & (pl.col("luck_reb") >= LUCK_TAG))
                     | ((pl.col("stat").str.contains("ast")) & (pl.col("luck_ast") >= LUCK_TAG)))
               .then(pl.lit("luck"))
               .when(pl.col("season_games_before") < 10).then(pl.lit("new_or_early"))
               .otherwise(pl.lit("other")).alias("angle")))
    out.write_parquet(OUT)
    print(f"wrote {OUT}: {out.height:,} rungs, {out['event_id'].n_unique()} games")
    return out


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def information_test(df: pl.DataFrame) -> dict:
    """Logistic regression hit ~ a + b*logit(p_book) + c*logit(p_model). c > 0 = model adds info."""
    from scipy.optimize import minimize

    y = df["hit"].to_numpy().astype(float)
    xb, xm = _logit(df["p_book"].to_numpy()), _logit(df["p_model"].to_numpy())

    def fit(cols):
        x = np.column_stack([np.ones_like(y), *cols])

        def nll(w):
            z = x @ w
            return np.mean(np.logaddexp(0, z) - y * z)
        r = minimize(nll, np.zeros(x.shape[1]), method="BFGS")
        se = np.sqrt(np.diag(r.hess_inv) / len(y))
        return r.x, se, r.fun

    _, _, llb = fit([xb])
    wbm, se, llbm = fit([xb, xm])
    _, _, llm = fit([xm])
    return {"n": len(y), "logloss_book_only": round(llb, 5), "logloss_model_only": round(llm, 5),
            "logloss_book+model": round(llbm, 5), "coef_book": round(wbm[1], 3),
            "coef_model": round(wbm[2], 3), "coef_model_se": round(se[2], 3)}


def flag(df: pl.DataFrame, cushions: tuple[float, float, float]) -> pl.DataFrame:
    hi, mid, lo = cushions
    ev = pl.col("price") * pl.col("p_model") - 1
    cush = (pl.when(pl.col("p_model") >= 0.40).then(hi)
            .when(pl.col("p_model") >= 0.20).then(mid).otherwise(lo))
    return df.with_columns(ev.alias("ev")).filter(
        (pl.col("ev") >= cush) & (pl.col("ev") < CHECK_NEWS_EV) & (1 / pl.col("p_model") <= MAX_FAIR))


def roi(bets: pl.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    if bets.is_empty():
        return {"bets": 0}
    pnl = np.where(bets["hit"].to_numpy(), bets["price"].to_numpy() - 1, -1.0)
    rng = np.random.default_rng(seed)
    # resample by game day (bets on the same night are correlated)
    days = bets["game_date"].to_numpy()
    uniq = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    boots = [np.mean(np.concatenate([pnl[idx[d]] for d in rng.choice(uniq, len(uniq))]))
             for _ in range(n_boot)]
    return {"bets": len(pnl), "hit_rate": round(float(bets["hit"].mean()), 3),
            "avg_price": round(float(bets["price"].mean()), 2), "roi": round(float(pnl.mean()), 4),
            "roi_ci90": (round(float(np.percentile(boots, 5)), 4), round(float(np.percentile(boots, 95)), 4))}


def report() -> None:
    df = pl.read_parquet(OUT)
    days = sorted(df["game_date"].unique().to_list())
    random.Random(1).shuffle(days)
    tune_days = set(days[: len(days) // 2])
    tune = df.filter(pl.col("game_date").is_in(list(tune_days)))
    test = df.filter(~pl.col("game_date").is_in(list(tune_days)))
    print(f"{df.height:,} rungs, {df['event_id'].n_unique()} games, {len(days)} days "
          f"(tune {tune['game_date'].n_unique()} / test {test['game_date'].n_unique()} days)")
    print("\n1) Does the model add information beyond Sportsbet's price? (all rungs)")
    print("  ", information_test(df))
    grid = [(a, b, c) for a in (0.05, 0.10, 0.15) for b in (0.10, 0.15, 0.25) for c in (0.20, 0.30, 0.50)]
    scored = sorted(((roi(flag(tune, g))["roi"] if flag(tune, g).height >= 30 else -9, g) for g in grid),
                    reverse=True)
    best = scored[0][1]
    print(f"\n2) Value bets at Sportsbet price. Cushions (p>=40%, 20-40%, <20%) tuned on half: {best}")
    print("   tune half:", roi(flag(tune, best)))
    print("   TEST half:", roi(flag(test, best)))
    print("   test half, default cushions (10/15/25%):", roi(flag(test, (0.10, 0.15, 0.25))))
    print("\n3) TEST half by angle:")
    for ang in ["starter_out", "luck", "new_or_early", "other"]:
        print(f"   {ang:13s}", roi(flag(test.filter(pl.col("angle") == ang), best)))
    print("\n   TEST half by stat:")
    for s in sorted(df["stat"].unique().to_list()):
        print(f"   {s:13s}", roi(flag(test.filter(pl.col("stat") == s), best)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "report"])
    args = ap.parse_args()
    build() if args.cmd == "build" else report()


if __name__ == "__main__":
    main()
