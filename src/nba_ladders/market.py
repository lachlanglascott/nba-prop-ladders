"""Market vs model: Sportsbet prices next to our fair prices, flagged when worth taking.

    uv run python -m nba_ladders.market                      # latest snapshot, all priced players
    uv run python -m nba_ladders.market --player "Moussa Diabaté"
    uv run python -m nba_ladders.market --min-ev 0 --all     # every rung, not only value

Reads the latest cached Odds API snapshot per event (take one first:
`python -m nba_ladders.sources.odds_api snapshot`). For each player Sportsbet prices:
- model probability for each rung (single stats, and P+R / P+A / R+A *totals*),
- Sportsbet's price and EV = price x p_model - 1,
- US reference: the US books' price with their margin removed (margin estimated from their
  Over/Under main lines, applied to the one-sided alternate ladders).
A rung is flagged VALUE when EV clears a cushion that grows with the odds (model uncertainty
is bigger in the tails): 10% if p >= 40%, 15% if 20-40%, 25% below 20%, and only if fair <= $15.
EV >= +100% is flagged CHECK NEWS instead: the book almost certainly knows something
(injury, role change) the model doesn't.
"""

import argparse

import numpy as np
import polars as pl
from nba_api.stats.static import teams

from nba_ladders.price import _frame, simulate_player
from nba_ladders.sources.injury_report import player_key
from nba_ladders.sources.odds_api import AU_BASE, US_REFERENCE, load_cached

TEAM_ABBR = {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}
TEAM_ABBR["LA Clippers"] = "LAC"


MAX_FAIR = 15.0  # model isn't validated deeper in the tail: never flag rungs past this
CHECK_NEWS_EV = 1.0  # EV this big means the book knows something (injury, role): don't trust it


def cushion(p: float) -> float:
    return 0.10 if p >= 0.40 else 0.15 if p >= 0.20 else 0.25


def latest(df: pl.DataFrame) -> pl.DataFrame:
    """Keep each event's most recent snapshot."""
    last = df.group_by("event_id").agg(pl.col("snapshot").max())
    return df.join(last, on=["event_id", "snapshot"])


def us_reference(df: pl.DataFrame) -> pl.DataFrame:
    """No-vig US probability per (player, stat, threshold): median over US books.

    Each book's overround comes from its two-sided main lines; alternates are one-sided, so
    implied probability is divided by that book's average overround.
    """
    us = df.filter(pl.col("bookmaker").is_in(US_REFERENCE))
    main = us.filter(~pl.col("alternate"), pl.col("point").is_not_null())
    pairs = main.pivot(on="side", index=["bookmaker", "player", "stat", "point"], values="price",
                       aggregate_function="first")
    if not {"Over", "Under"} <= set(pairs.columns):
        return pl.DataFrame(schema={"player": pl.String, "stat": pl.String, "threshold": pl.Int64,
                                    "p_us": pl.Float64})
    over = (pairs.drop_nulls(["Over", "Under"])
            .with_columns((1 / pl.col("Over") + 1 / pl.col("Under")).alias("overround"))
            .group_by("bookmaker").agg(pl.col("overround").median()))
    return (us.filter(pl.col("side") == "Over", pl.col("threshold").is_not_null())
            .join(over, on="bookmaker", how="left")
            .with_columns(((1 / pl.col("price")) / pl.col("overround").fill_null(1.07)).alias("p"))
            .group_by("player", "stat", "threshold").agg(pl.col("p").median().alias("p_us")))


def model_p(stats: dict, stat: str, t: int) -> float:
    values = sum(stats[s] for s in stat.split("+"))
    return float(np.mean(values >= t))


def value_report(df: pl.DataFrame, min_ev: float | None = None, player: str | None = None,
                 show_all: bool = False) -> pl.DataFrame:
    df = latest(df)
    sb = df.filter(pl.col("bookmaker") == AU_BASE, pl.col("side") == "Over",
                   pl.col("threshold").is_not_null())
    if player:
        sb = sb.filter(pl.col("player") == player)
    ref = us_reference(df)
    names = _frame("player_games").sort("game_date").group_by("player_id").last().select(
        "player_id", "player_name", "team").with_columns(
        pl.col("player_name").map_elements(player_key, return_dtype=pl.String).alias("key"))
    rows = []
    for (who, home, away), g in sb.group_by(["player", "home_team", "away_team"]):
        match = names.filter(pl.col("key") == player_key(who))
        if match.is_empty():
            continue
        m = match.row(0, named=True)
        h, a = TEAM_ABBR.get(home), TEAM_ABBR.get(away)
        opp = a if m["team"] == h else h
        sim = simulate_player(m["player_name"], opponent=opp, verbose=False, n=40_000)
        if sim is None:
            continue
        stats = sim[0]
        for r in g.unique(["stat", "threshold"]).iter_rows(named=True):
            p = model_p(stats, r["stat"], r["threshold"])
            if p <= 0:
                continue
            rows.append({"player": m["player_name"], "team": m["team"], "opp": opp,
                         "line": f"{r['threshold']}+ {r['stat']}", "stat": r["stat"],
                         "threshold": r["threshold"], "sportsbet": r["price"], "p_model": p,
                         "fair": 1 / p, "ev": r["price"] * p - 1, "cushion": cushion(p),
                         "player_api": who})
    if not rows:
        return pl.DataFrame()
    out = (pl.DataFrame(rows)
           .join(ref.rename({"player": "player_api"}), on=["player_api", "stat", "threshold"], how="left")
           .with_columns((1 / pl.col("p_us")).alias("us_fair"))
           .with_columns(
               pl.when(pl.col("ev") >= CHECK_NEWS_EV).then(pl.lit("CHECK NEWS"))
               .when((pl.col("ev") >= pl.col("cushion")) & (pl.col("fair") <= MAX_FAIR))
               .then(pl.lit("VALUE")).otherwise(pl.lit("")).alias("flag"))
           .with_columns((pl.col("flag") == "VALUE").alias("value")))
    if not show_all:
        out = out.filter(pl.col("flag") != "")
    if min_ev is not None:
        out = out.filter(pl.col("ev") >= min_ev)
    return out.drop("player_api", "stat", "threshold", "cushion").sort("ev", descending=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player")
    ap.add_argument("--min-ev", type=float)
    ap.add_argument("--all", action="store_true", help="show every rung, not only flagged value")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    df = load_cached("live")
    if df.is_empty():
        raise SystemExit("no cached snapshot; run: python -m nba_ladders.sources.odds_api snapshot")
    print(f"snapshot(s): {sorted(latest(df)['snapshot'].unique().to_list())[-3:]}")
    rep = value_report(df, args.min_ev, args.player, args.all)
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=args.top + 2, tbl_width_chars=170,
                   fmt_str_lengths=30):
        print(rep.head(args.top).with_columns(
            pl.col("p_model", "fair", "us_fair", "sportsbet").round(2), pl.col("ev").round(3)))


if __name__ == "__main__":
    main()
