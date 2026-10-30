"""Opportunity vs outcome: potential assists and rebound chances as a guide to future production.

"8 potential assists but only 2 actual" is bad luck if conversion is mostly noise around a
player's long-run rate. Test: predict a player's next-game assists (rebounds) from his last N
games using (a) actual per-minute output, (b) expected output = chances per minute x his
shrunk conversion rate, (c) a blend. Predictions use the next game's actual minutes, so only
the per-minute rate is being judged.

    uv run python -m nba_ladders.luck                     # predictive test, 2024-25 + 2025-26
    uv run python -m nba_ladders.luck "Deni Avdija"       # a player's recent luck
    uv run python -m nba_ladders.luck --scan [YYYY-MM-DD] # a game day's luck candidates
"""

import argparse
from functools import lru_cache

import polars as pl

from nba_ladders import config

WINDOW = 10
K_CONV = {"ast": 60.0, "reb": 80.0}  # pseudo-chances of league conversion in the shrink
CHANCES = {"ast": "potential_ast", "reb": "reb_chances"}
# Weight on the chance-based rate when blending with the actual recent rate.
# AST 0.5: best on 2024-25 and again on 2025-26. REB: 2024-25 preferred 0.7 but 2025-26 0.2,
# so a cautious 0.3 (chosen after seeing both seasons; treat as provisional). docs/luck.md
BLEND = {"ast": 0.5, "reb": 0.3}


def load() -> pl.DataFrame:
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    trk = pl.read_parquet(config.DATA_PROCESSED / "tracking.parquet").drop("ast")
    return pg.join(trk, on=["player_id", "game_date"]).sort("player_id", "game_date")


def league_conversion(df: pl.DataFrame) -> dict[str, float]:
    return {s: df[s].sum() / df[c].sum() for s, c in CHANCES.items()}


def rolling_features(df: pl.DataFrame, league: dict[str, float], window: int = WINDOW) -> pl.DataFrame:
    """Per player-game, features from strictly earlier games (as-at)."""
    def prior_sum(col: str, n: int | None = None) -> pl.Expr:
        c = pl.col(col).shift(1).over("player_id")
        return (c.rolling_sum(n, min_samples=1) if n else c.cum_sum()).over("player_id")

    feats = [prior_sum("minutes", window).alias("w_min"), pl.col("player_id").cum_count().over("player_id").alias("n_prior")]
    for s, c in CHANCES.items():
        k = K_CONV[s]
        feats += [
            prior_sum(s, window).alias(f"w_{s}"),
            prior_sum(c, window).alias(f"w_{c}"),
            # long-run conversion from all earlier games, shrunk to league
            ((prior_sum(s) + k * league[s]) / (prior_sum(c) + k)).alias(f"conv_{s}"),
        ]
    out = df.with_columns(feats)
    preds = []
    for s, c in CHANCES.items():
        actual_rate = pl.col(f"w_{s}") / pl.col("w_min")
        expected_rate = pl.col(f"w_{c}") / pl.col("w_min") * pl.col(f"conv_{s}")
        preds += [
            (actual_rate * pl.col("minutes")).alias(f"pred_{s}_actual"),
            (expected_rate * pl.col("minutes")).alias(f"pred_{s}_expected"),
            ((actual_rate + expected_rate) / 2 * pl.col("minutes")).alias(f"pred_{s}_blend"),
            # luck over the window: actual minus expected (negative = unlucky)
            (pl.col(f"w_{s}") - pl.col(f"w_{c}") * pl.col(f"conv_{s}")).alias(f"luck_{s}"),
        ]
    return out.with_columns(preds)


def predictive_test(feat: pl.DataFrame, seasons=("22024", "22025")) -> pl.DataFrame:
    test = feat.filter(
        pl.col("season_id").is_in(list(seasons)), pl.col("n_prior") >= WINDOW,
        pl.col("w_min") >= 100, pl.col("minutes") >= 10,
    )
    rows = []
    for s in CHANCES:
        for kind in ("actual", "expected", "blend"):
            err = pl.col(f"pred_{s}_{kind}") - pl.col(s)
            rows.append(test.select(
                pl.lit(s).alias("stat"), pl.lit(kind).alias("predictor"), pl.len().alias("n"),
                (err ** 2).mean().sqrt().alias("rmse"), err.abs().mean().alias("mae"),
                pl.corr(f"pred_{s}_{kind}", s).alias("corr"),
            ))
    return pl.concat(rows)


def luck_buckets(feat: pl.DataFrame, stat: str, seasons=("22024", "22025")) -> pl.DataFrame:
    """Next-game actual vs 'actual-rate' prediction, by how lucky the last WINDOW games were."""
    test = feat.filter(pl.col("season_id").is_in(list(seasons)), pl.col("n_prior") >= WINDOW,
                       pl.col("w_min") >= 100, pl.col("minutes") >= 10)
    return (
        test.with_columns(
            (pl.col(f"luck_{stat}") / pl.col(f"w_{CHANCES[stat]}").clip(lower_bound=1))
            .qcut(5, labels=["very unlucky", "unlucky", "neutral", "lucky", "very lucky"])
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.len().alias("games"),
            pl.col(stat).mean().alias("next_actual"),
            pl.col(f"pred_{stat}_actual").mean().alias("pred_from_actual"),
            pl.col(f"pred_{stat}_expected").mean().alias("pred_from_chances"),
        )
        .sort("bucket")
    )


def player_report(feat: pl.DataFrame, name: str, last: int = 10) -> None:
    p = feat.filter(pl.col("player_name") == name).tail(last)
    if p.is_empty():
        raise SystemExit(f"no tracking rows for {name!r}")
    conv_a, conv_r = p["conv_ast"][-1], p["conv_reb"][-1]
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=last + 2, tbl_width_chars=160):
        print(f"{name}: last {p.height} tracked games. Long-run conversion: "
              f"AST {conv_a:.0%} of potential, REB {conv_r:.0%} of chances")
        print(p.select(
            "game_date", "matchup", "minutes", "ast", "potential_ast",
            (pl.col("potential_ast") * conv_a).round(1).alias("exp_ast"),
            "reb", "reb_chances", (pl.col("reb_chances") * conv_r).round(1).alias("exp_reb"),
        ))
        for s, c, conv in (("ast", "potential_ast", conv_a), ("reb", "reb_chances", conv_r)):
            act, exp = p[s].sum(), p[c].sum() * conv
            adj = (BLEND[s] * exp + (1 - BLEND[s]) * act) / act if act else float("nan")
            print(f"  {s.upper()}: {act} actual vs {exp:.1f} expected over these games "
                  f"({act - exp:+.1f}, {'unlucky' if act < exp else 'lucky'}); "
                  f"luck-adjusted rate x{adj:.2f} vs recent actual")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("player", nargs="?")
    ap.add_argument("--scan", nargs="?", const="today", metavar="YYYY-MM-DD",
                    help="scan a game day's players for luck gaps (default: today ET)")
    args = ap.parse_args()
    if args.scan:
        from datetime import date as _date
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo

        day = (_dt.now(ZoneInfo("America/New_York")).date() if args.scan == "today"
               else _date.fromisoformat(args.scan))
        _cli_scan(day)
        return
    df = load()
    feat = rolling_features(df, league_conversion(df))
    if args.player:
        player_report(feat, args.player)
        return
    lg = league_conversion(df)
    print(f"league conversion: AST {lg['ast']:.1%} of potential, REB {lg['reb']:.1%} of chances")
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=20):
        print(predictive_test(feat).with_columns(pl.col(pl.Float64).round(3)))
        for s in CHANCES:
            print(f"{s}: next game by luck over the last {WINDOW}")
            print(luck_buckets(feat, s).with_columns(pl.col(pl.Float64).round(2)))



K_LUCK = {"ast": 25.0, "reb": 40.0}  # pseudo-chances: small samples barely move the price


def luck_multipliers(current: pl.DataFrame, history: pl.DataFrame,
                     league: dict[str, float]) -> pl.DataFrame:
    """Per player: rate multipliers for AST / REB from this season's luck so far.

    current: season_totals() rows (ast, potential_ast, reb, reb_chances per player).
    history: past per-game tracking (load()) for each player's long-run conversion.
    mult = blended / actual, where blended = BLEND * chances x conv + (1 - BLEND) * actual,
    then shrunk toward 1 by sample size (chances / (chances + K_LUCK)).
    """
    conv = history.group_by("player_id").agg(
        *[pl.col(s).sum().alias(f"h_{s}") for s in CHANCES],
        *[pl.col(c).sum().alias(f"h_{c}") for c in CHANCES.values()],
    )
    d = current.join(conv, on="player_id", how="left").fill_null(0)
    out = []
    for s, c in CHANCES.items():
        k = K_CONV[s]
        player_conv = (pl.col(f"h_{s}") + pl.col(s) + k * league[s]) / (pl.col(f"h_{c}") + pl.col(c) + k)
        expected = pl.col(c) * player_conv
        blended = BLEND[s] * expected + (1 - BLEND[s]) * pl.col(s)
        raw = (blended / pl.col(s).clip(lower_bound=1)).clip(0.7, 1.6)
        weight = pl.col(c) / (pl.col(c) + K_LUCK[s])
        out.append((1 + weight * (raw - 1)).alias(f"luck_{s}"))
    return d.select("player_id", *out)


@lru_cache(maxsize=8)
def live_luck(season: str = "2026-27", as_of=None) -> pl.DataFrame:
    """Current-season luck multipliers for every player (snapshots tracking totals today).

    as_of: replay a past ET game day from that day's snapshot (raises FileNotFoundError).
    """
    from nba_ladders.sources.tracking import season_totals

    hist = load()
    if as_of is not None:
        hist = hist.filter(pl.col("game_date") < as_of)
    return luck_multipliers(season_totals(season, as_of=as_of), hist, league_conversion(hist))


def effective_luck(mult: float, weights, current_season) -> float:
    """Apply a season-to-date luck multiplier only to the current season's share of the fit.

    The multiplier corrects *this season's* actual rate toward its expected rate. The simulator
    samples a recency-weighted mix of this season and older games; older games carry no recent
    luck, so scaling all of them would double-count the reversion.
    weights: template weights (None = equal); current_season: bool mask per template game.
    """
    import numpy as np

    cur = np.asarray(current_season, dtype=bool)
    if cur.size == 0:
        return 1.0
    w = np.ones(cur.size) if weights is None else np.asarray(weights, dtype=float)
    share = float(w[cur].sum() / w.sum())
    return 1 + (mult - 1) * share


SCAN_MIN_MPG = 20.0
SCAN_MIN_CHANCES = {"ast": 20, "reb": 25}


def scan(report: pl.DataFrame, game_day, top: int = 6) -> dict[str, pl.DataFrame]:
    """Players in game_day's games whose season-to-date output lags their chances most.

    report: parsed injury report with a 'team' abbreviation column. Excludes Out/Doubtful.
    Returns {"ast": df, "reb": df} with actual vs expected per game and the luck multiplier.
    """
    from nba_ladders.screens import parse_player_index
    from nba_ladders.sources.injury_report import player_key
    from nba_ladders.sources.nba_stats import get_player_index
    from nba_ladders.sources.tracking import season_totals

    day = report.filter(pl.col("game_date") == game_day)
    games = day.select("matchup", "team").unique().drop_nulls()
    out_keys = day.filter(pl.col("status").is_in(["Out", "Doubtful"]))["player_key"].drop_nulls()
    idx = parse_player_index(get_player_index("2026-27")).with_columns(
        pl.col("player_name").map_elements(player_key, return_dtype=pl.String).alias("key"))
    hist = load()
    lg = league_conversion(hist)
    conv = hist.group_by("player_id").agg(
        *[pl.col(s).sum().alias(f"h_{s}") for s in CHANCES],
        *[pl.col(c).sum().alias(f"h_{c}") for c in CHANCES.values()])
    d = (season_totals("2026-27").join(idx, on="player_id").join(games, left_on="team_now", right_on="team")
         .join(conv, on="player_id", how="left").fill_null(0)
         .filter(~pl.col("key").is_in(out_keys.implode()))
         .with_columns((pl.col("min") / pl.col("gp")).alias("mpg"))
         .filter(pl.col("mpg") >= SCAN_MIN_MPG)
         .join(live_luck(), on="player_id"))
    res = {}
    for s, c in CHANCES.items():
        k = K_CONV[s]
        player_conv = (pl.col(f"h_{s}") + k * lg[s]) / (pl.col(f"h_{c}") + k)
        t = (d.filter(pl.col(c) >= SCAN_MIN_CHANCES[s])
             .with_columns((pl.col(c) * player_conv).alias("exp"))
             .with_columns(((pl.col("exp") - pl.col(s)) / pl.col("gp")).alias("short_pg"))
             .sort("short_pg", descending=True).head(top))
        res[s] = t.select(
            "player_id", "player_name", "team_now", "matchup", "gp", pl.col("mpg").round(1),
            (pl.col(s) / pl.col("gp")).round(1).alias(f"{s}_pg"),
            (pl.col(c) / pl.col("gp")).round(1).alias("chances_pg"),
            (pl.col("exp") / pl.col("gp")).round(1).alias("exp_pg"),
            (pl.col(s) / pl.col(c)).round(2).alias("conv_now"), player_conv.round(2).alias("conv_career"),
            pl.col(f"luck_{s}").round(2))
    return res


def _cli_scan(game_day) -> None:
    from nba_api.stats.static import teams

    from nba_ladders.sources.injury_report import report_for_date

    abbr = {t["full_name"].replace(" ", ""): t["abbreviation"] for t in teams.get_teams()}
    rep = report_for_date(game_day).with_columns(
        pl.col("team_name").replace_strict(abbr, default=None).alias("team"))
    res = scan(rep, game_day, top=10)
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=12, tbl_cols=14, tbl_width_chars=200):
        for s in CHANCES:
            print(f"\n{s.upper()}: furthest below expected (season to date), {game_day}")
            print(res[s].drop("player_id"))


if __name__ == "__main__":
    main()
