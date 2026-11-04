"""Candidates report: every angle's candidates for a game day, with evidence, prediction, prices.

    uv run python -m nba_ladders.candidates                 # today's ET slate
    uv run python -m nba_ladders.candidates --date 2026-10-30

Angles find the candidates (the user's edges): potentials (last-game misses, volume spikes,
season luck), starter out (backup bigs), minutes unlocked, Questionable teammates, rebound
chances, foul spot, high-variance backups. Bets are then the model's projection vs the market:
fair = 1 / model probability, EV at the best AU price, focus on rungs priced $3-$15.

Per candidate: reasons, prediction (minutes, means, tracking volume and per-game variance),
ladder prices (Sportsbet, best AU, fair, EV), and SGM combos with fair prices for comparing
against the SGM builder. Needs a same-day odds snapshot for prices
(`python -m nba_ladders.sources.odds_api snapshot`). Writes data/processed/sheets/.
"""

import argparse
import re
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from nba_ladders import config
from nba_ladders.absences import (
    MIN_MATE_MINUTES,
    RATES,
    context_effects,
    pair_effects,
    priors,
    shrunk_effects,
)
from nba_ladders.combos import best_combos
from nba_ladders.context_adjust import adjust_history, history_adjustment
from nba_ladders.luck import load as load_tracking
from nba_ladders.luck import scan
from nba_ladders.market import TEAM_ABBR, latest, model_p
from nba_ladders.price import _frame, simulate_player
from nba_ladders.scenarios import positions
from nba_ladders.screens import parse_player_index
from nba_ladders.sheet import P_OUT, depth_chart, is_new_absence
from nba_ladders.sources.injury_report import ET, player_key, report_for_date
from nba_ladders.sources.nba_stats import get_player_index
from nba_ladders.sources.odds_api import AU_BASE, AU_REFERENCE, load_cached

AU_BOOKS = [AU_BASE, *AU_REFERENCE]
UNLOCK_MIN = 3.0
Q_WATCH_MIN = 2.0  # a Questionable teammate's absence worth flagging (effects are shrunk/scaled)
FOUL_Q = 0.667  # top third of starter fouls/36 x opponent FTA (last season)
HI_VAR_MIN_SD = 5.8  # top tercile of minutes sd

EVIDENCE = {  # what each angle means (no backtest numbers: bets are model vs market)
    "pot_volume_ast": "A potentials spike last game with tonight's lineup: his role may be bigger than his average.",
    "pot_miss_ast": "Monster potentials last game that didn't convert, with tonight's lineup.",
    "starter_out": "Starting centre out: backup big's minutes and rebound role jump.",
    "unlocked": "Minutes freed by tonight's absences.",
    "luck_ast": "Season potentials ahead of his assists: conversion should come back toward his norm.",
    "luck_reb": "Rebound chances ahead of his rebounds.",
    "foul_spot": "Foul-prone starter vs a high-FTA opponent: extra backup minutes if he gets in trouble.",
    "hi_var_backup": "High-variance backup minutes: tails are where ladders pay.",
    "model_scan": "No angle: in the slate because the model's price beats the market.",
    "q_watch": "Questionable teammate: priced as a sits/plays mix; if he's ruled out, use the 'if he sits' column.",
}
FOCUS = {"model_scan": ["pts", "reb", "ast", "pts+reb", "pts+ast", "reb+ast"], "pot_miss_ast": ["ast"], "pot_volume_ast": ["ast"], "starter_out": ["reb", "pts+reb", "pts"], "unlocked": ["pts", "reb", "pts+reb", "ast"],
         "luck_ast": ["ast"], "luck_reb": ["reb", "pts+reb"], "foul_spot": ["reb", "pts+reb"],
         "hi_var_backup": ["reb", "pts+reb"], "q_watch": ["pts", "reb", "pts+reb", "ast"]}
SHAPES = {"model_scan": ["p/r", "p/a", "r/a"], "pot_miss_ast": ["p/a", "r/a"], "pot_volume_ast": ["p/a", "r/a"], "starter_out": ["p/r", "p/r/a"], "unlocked": ["p/r", "p/a", "p/r/a"],
          "luck_ast": ["p/a", "r/a"], "luck_reb": ["p/r", "r/a"], "foul_spot": ["p/r"],
          "hi_var_backup": ["p/r", "p/r/a"], "q_watch": ["p/r", "p/a", "p/r/a"]}


def _opp(matchup: str, team: str) -> str:
    away, home = matchup.split("@")
    return away if team == home else home


def _recent(pg: pl.DataFrame, n: int = 20) -> pl.DataFrame:
    s = pg.sort("game_date").group_by("player_id").tail(n)
    return s.group_by("player_id").agg(
        pl.col("minutes").mean().alias("min20"), pl.col("minutes").std().alias("min_sd"),
        (pl.col("pf").sum() / pl.col("minutes").sum() * 36).alias("pf36"),
        pl.col("game_date").max().alias("last_game"))


def _team_fta(pg: pl.DataFrame, n: int = 20) -> dict[str, float]:
    t = pg.group_by("game_id", "team", "game_date").agg(pl.col("fta").sum()).sort("game_date")
    return dict(t.group_by("team").tail(n).group_by("team").agg(pl.col("fta").mean()).iter_rows())


def _foul_threshold(pg: pl.DataFrame) -> float:
    """Top-third cut of starter pf36 x opponent FTA over last season's backup-big games."""
    bg = _frame("backup_big_games").filter(pl.col("season_id") == "22025")
    s = pg.filter(pl.col("season_id") == "22025")
    pf = s.group_by("player_id").agg((pl.col("pf").sum() / pl.col("minutes").sum() * 36).alias("pf36"))
    fta = s.group_by("game_id", "team").agg(pl.col("fta").sum()).group_by("team").agg(pl.col("fta").mean())
    x = (bg.join(pf.rename({"player_id": "starter_id"}), on="starter_id")
         .join(fta.rename({"team": "opponent"}), on="opponent"))
    return float((x["pf36"] * x["fta"]).quantile(FOUL_Q))


def gather(day: date) -> tuple[dict, pl.DataFrame, dict]:
    """Candidates keyed by player name -> {team, opp, reasons{angle: detail}, scenario}."""
    report = report_for_date(day).filter(pl.col("game_date") == day).with_columns(
        pl.col("team_name").replace_strict({k.replace(" ", ""): v for k, v in TEAM_ABBR.items()},
                                           default=None).alias("team"))
    pg = _frame("player_games")
    report = add_b2b_rest(report, pg, day)
    report = add_carry_forward(report, pg, day)
    pos = positions()
    roster = parse_player_index(get_player_index("2026-27"))
    dc = depth_chart(pg, roster, pos)
    status = report.select("team", "player_key", "status", "reason")
    st = lambda team, key: status.filter((pl.col("team") == team) & (pl.col("player_key") == key))
    matchups = dict(report.select("team", "matchup").unique().drop_nulls().iter_rows())
    out_keys = set(report.filter(pl.col("status").is_in(["Out", "Doubtful"]))["player_key"].drop_nulls())
    recent = _recent(pg)
    cutoff = pg["game_date"].max() - timedelta(days=45)
    fta = _team_fta(pg)
    foul_q = _foul_threshold(pg)
    cands: dict[str, dict] = {}
    qwatch: dict[str, list[str]] = {}

    def add(name, team, angle, detail, strength: float = 0.0, **scenario):
        c = cands.setdefault(name, {"team": team, "opp": _opp(matchups[team], team), "reasons": {},
                                    "scenario": {}, "strength": {}})
        c["reasons"][angle] = detail
        c["strength"][angle] = strength
        c["scenario"].update(scenario)

    for team in sorted(matchups):
        chart = dc.filter(pl.col("team_now") == team).join(recent.select("player_id", "min_sd", "pf36", "last_game"),
                                                           on="player_id", how="left")
        starter = chart.filter(pl.col("is_starter"))
        if starter.is_empty():
            continue
        s = starter.row(0, named=True)
        ss = st(team, s["key"])
        s_status = ss["status"][0] if ss.height else "Not listed"
        backups = chart.filter(~pl.col("is_starter"), ~pl.col("key").is_in(list(out_keys)),
                               pl.col("last_game") >= cutoff)
        for b in backups.head(2).iter_rows(named=True):
            if s_status in ("Out", "Doubtful", "Questionable"):
                new = is_new_absence(pg, s["player_id"], team, day)
                if new:
                    add(b["player_name"], team, "starter_out",
                        f"{s['player_name']} {s_status} (P(sits) {P_OUT[s_status]:.0%}), **NEW** (played last game)",
                        strength=P_OUT[s_status], p_out=P_OUT[s_status])
                else:  # ongoing absence: his recent minutes already reflect it; price his current role
                    add(b["player_name"], team, "starter_out",
                        f"{s['player_name']} {s_status}, ongoing absence (already in his recent role and the price)")
            elif s.get("pf36") and fta.get(_opp(matchups[team], team)):
                score = s["pf36"] * fta[_opp(matchups[team], team)]
                if score >= foul_q:
                    add(b["player_name"], team, "foul_spot",
                        f"{s['player_name']} {s['pf36']:.1f} fouls/36 vs opponent "
                        f"{fta[_opp(matchups[team], team)]:.1f} FTA/g")
            if (b["min_sd"] or 0) >= HI_VAR_MIN_SD:
                add(b["player_name"], team, "hi_var_backup",
                    f"minutes sd {b['min_sd']:.1f} over last 20 (avg {b['recent_min']:.0f})")

    # minutes unlocked by tonight's Out/Doubtful teammates who played recently
    eff_raw = pair_effects(pg, pos)
    pri = priors(eff_raw)
    eff = shrunk_effects(eff_raw, pri)
    people = roster.join(recent, on="player_id").with_columns(
        pl.col("player_name").map_elements(player_key, return_dtype=pl.String).alias("player_key"))
    absent = (report.filter(pl.col("status").is_in(["Out", "Doubtful"]))
              .join(people, left_on=["team", "player_key"], right_on=["team_now", "player_key"])
              .filter(pl.col("min20") >= MIN_MATE_MINUTES, pl.col("last_game") >= cutoff)
              .select(pl.col("team").alias("team_id"), pl.col("player_id").alias("mate_id"),
                      pl.col("min20").alias("mate_min"), pl.col("player_name").alias("mate"),
                      pl.lit("t").alias("game_id")))
    playing = people.filter(pl.col("team_now").is_in(list(matchups)), ~pl.col("player_key").is_in(list(out_keys)),
                            pl.col("last_game") >= cutoff).select(
        "player_id", "player_name", pl.col("team_now").alias("team_id"), pl.lit("t").alias("game_id"))
    ctx = context_effects(playing, absent.drop("mate"), eff, pri, pos).join(playing.drop("game_id"), on="player_id")
    for r in ctx.filter(pl.col("shift_min") >= UNLOCK_MIN).iter_rows(named=True):
        who = absent.filter(pl.col("team_id") == r["team_id"])["mate"].to_list()
        add(r["player_name"], r["team_id"], "unlocked", f"+{r['shift_min']:.1f} min; out: {', '.join(who)}",
            strength=r["shift_min"])

    # Questionable watch: who gains if a Questionable rotation player sits (hypothetical absence)
    q = (report.filter(pl.col("status") == "Questionable", ~pl.col("reason").str.contains("(?i)g ?league|two-way"))
         .join(people, left_on=["team", "player_key"], right_on=["team_now", "player_key"])
         .filter(pl.col("min20") >= MIN_MATE_MINUTES, pl.col("last_game") >= cutoff))
    for qr in q.iter_rows(named=True):
        hypo = pl.concat([absent.filter(pl.col("team_id") == qr["team"]).drop("mate"),
                          pl.DataFrame({"team_id": [qr["team"]], "mate_id": [qr["player_id"]],
                                        "mate_min": [qr["min20"]], "game_id": ["t"]})], how="vertical_relaxed")
        mates = playing.filter(pl.col("team_id") == qr["team"], pl.col("player_id") != qr["player_id"])
        base = ctx.filter(pl.col("team_id") == qr["team"]).select("player_id", pl.col("shift_min").alias("base"))
        hyp = context_effects(mates, hypo, eff, pri, pos).join(mates.drop("game_id"), on="player_id") \
            .join(base, on="player_id", how="left").with_columns(pl.col("base").fill_null(0.0)) \
            .with_columns((pl.col("shift_min") - pl.col("base")).alias("gain"))
        if hyp.height and qr["min20"] >= 20:
            top = hyp.sort("gain", descending=True).head(3)
            qwatch.setdefault(qr["team"], []).append(
                f"{qr['player_name']} Questionable ({qr['reason']}, {qr['min20']:.0f} min/g): if he sits, "
                + ", ".join(f"{x['player_name']} +{x['gain']:.1f} min" for x in top.iter_rows(named=True)))
        for r in hyp.filter(pl.col("gain") >= Q_WATCH_MIN).iter_rows(named=True):
            add(r["player_name"], r["team_id"], "q_watch",
                f"{qr['player_name']} Questionable ({qr['reason']}"
                + ("" if "P(sits)" in qr["reason"] else f", P(sits) {p_sits(qr):.0%}") + f"); if he sits: +{r['gain']:.1f} min",
                strength=r["gain"], q_minutes=r["gain"], q_mult={s: r[f"mult_{s}"] for s in RATES}, q_p=p_sits(qr))

    # potentials luck (season to date)
    lk = scan(report, day, top=8)
    for stat, angle in (("ast", "luck_ast"), ("reb", "luck_reb")):
        for r in lk[stat].iter_rows(named=True):
            if r[f"luck_{stat}"] >= 1.05 and r["chances_pg"] >= LUCK_MIN_CHANCES_PG[stat]:
                add(r["player_name"], r["team_now"], angle,
                    f"{r[f'{stat}_pg']}/g from {r['chances_pg']} chances (converting {r['conv_now']:.0%} vs "
                    f"{r['conv_career']:.0%} career); luck x{r[f'luck_{stat}']:.2f}",
                    strength=r[f"luck_{stat}"])
    # last 1-2 games of potentials, counted only when the lineup matched tonight's
    season_min = (pg.filter(pl.col("season_id") == "22026").group_by("player_id")
                  .agg(pl.col("minutes").mean().alias("mpg_now")))
    rot = people.join(season_min, on="player_id").filter(  # rotation = ROTATION_MPG+ this season
        pl.col("team_now").is_in(list(matchups)), pl.col("mpg_now") >= ROTATION_MPG)
    RECENT.clear()
    for r in playing.iter_rows(named=True):
        games = recent_games(pg, r["player_id"], r["team_id"], rot, out_keys)
        if not games:
            continue
        RECENT[r["player_name"]] = games
        vol = pot_volume(games[0], r["player_id"])
        if vol:
            add(r["player_name"], r["team_id"], "pot_volume_ast", vol["detail"], strength=vol["short"])
        miss = pot_miss(games[0], r["player_id"])
        if miss:
            add(r["player_name"], r["team_id"], "pot_miss_ast", miss["detail"], strength=miss["short"])
        for stat, angle in (("reb", "luck_reb"),):
            sig = recent_signal(games, stat, r["player_id"])
            if sig is None:
                continue
            note = sig["detail"]
            if angle in cands.get(r["player_name"], {}).get("reasons", {}):
                c = cands[r["player_name"]]
                c["reasons"][angle] += f"; {note}"
                c["strength"][angle] = max(c["strength"][angle], sig["luck"])
            else:
                add(r["player_name"], r["team_id"], angle, note, strength=sig["luck"])

    # model scan: every other 20+ mpg rotation player tonight gets priced (shown only if value)
    for r in rot.filter(~pl.col("player_key").is_in(list(out_keys))).iter_rows(named=True):
        if r["player_name"] not in cands:
            add(r["player_name"], r["team_now"], "model_scan", "no angle fired; priced because he's in the rotation")

    # tonight's absences (current teammates Out/Doubtful) for EVERY player: the history
    # adjustment backs out past absences of current teammates, so this adds tonight's back
    tonight_det = {r["player_name"]: (r["shift_min"], {s: r[f"mult_{s}"] for s in RATES})
                   for r in ctx.iter_rows(named=True)}
    # uncertain teammates (Questionable, B2B rest risk, carry-forward): weight their absence by P(sits)
    unc = q.select(pl.col("team").alias("team_id"), pl.col("player_id").alias("mate_id"), pl.col("min20").alias("mate_min"),
                   pl.struct("status", "p_sit").map_elements(lambda s: p_sits(s), return_dtype=pl.Float64).alias("p"),
                   pl.lit("t").alias("game_id"))
    tonight = dict(tonight_det)
    if unc.height:
        both = pl.concat([absent.drop("mate"), unc.drop("p")], how="vertical_relaxed")
        ctx_all = context_effects(playing, both, eff, pri, pos).join(playing.drop("game_id"), on="player_id")
        pbar = dict(unc.group_by("team_id").agg(pl.col("p").mean()).iter_rows())
        for r in ctx_all.filter(pl.col("team_id").is_in(list(pbar))).iter_rows(named=True):
            w = pbar[r["team_id"]]
            d_min, d_mult = tonight_det.get(r["player_name"], (0.0, {s: 1.0 for s in RATES}))
            tonight[r["player_name"]] = (d_min + w * (r["shift_min"] - d_min),
                                         {s: d_mult.get(s, 1.0) + w * (r[f"mult_{s}"] - d_mult.get(s, 1.0)) for s in RATES})
    from nba_ladders.sources.odds_api import latest_spreads

    abbr = {k: v for k, v in TEAM_ABBR.items()}
    spreads = {abbr.get(r["team_name"]): r["spread"] for r in latest_spreads().iter_rows(named=True)}
    min20 = dict(people.select("player_name", "min20").iter_rows())
    CONTEXT.update(day=day, pg=pg, eff=eff, pri=pri, pos=pos, roster=roster, tonight=tonight, tonight_det=tonight_det,
                   spreads=spreads, min20=min20,
                   avail={r["player_key"]: (r["status"], p_sits(r), r["reason"]) for r in report.filter(
                       pl.col("player_key").is_not_null()).iter_rows(named=True)})
    return cands, report, qwatch


def add_b2b_rest(report: pl.DataFrame, pg: pl.DataFrame, day: date) -> pl.DataFrame:
    """Second night of a back-to-back: players who usually rest it and aren't on the report
    become report rows (Out if they almost always sit, Questionable if it's a real risk)."""
    from nba_ladders.rest import SIT_OUT, tonight

    teams = report["team"].drop_nulls().unique().to_list()
    r = tonight(pg, day, teams)
    if r.is_empty():
        return report
    listed = set(report["player_key"].drop_nulls())
    games = dict(report.select("team", "matchup").unique().drop_nulls().iter_rows())
    meta = report.select("team", "team_name", "report_time", "game_date", "game_time").unique("team")
    rows = []
    for x in r.iter_rows(named=True):
        key = player_key(x["player_name"])
        if key in listed or x["team"] not in games:
            continue
        rows.append({"team": x["team"], "matchup": games[x["team"]], "player": x["player_name"], "player_key": key,
                     "status": "Out" if x["p_sit_b2b"] >= SIT_OUT else "Questionable", "p_sit": x["p_sit_b2b"],
                     "reason": f"B2B rest (not on report): sat {x['b2b_sat']} of {x['b2b_games']} second nights, "
                               f"P(sits) {x['p_sit_b2b']:.0%}"})
    if not rows:
        return report
    add = pl.DataFrame(rows).join(meta, on="team", how="left")
    if "p_sit" not in report.columns:
        report = report.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_sit"))
    return pl.concat([report, add.select(report.columns)], how="vertical_relaxed")


def add_carry_forward(report: pl.DataFrame, pg: pl.DataFrame, day: date) -> pl.DataFrame:
    """Players who missed their team's last game and aren't on tonight's report (see availability.py)."""
    from nba_ladders.availability import carry_forward

    if "p_sit" not in report.columns:
        report = report.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_sit"))

    def prev(d):
        try:
            r = report_for_date(d)
            return r.filter(pl.col("game_date") == d)
        except Exception:  # noqa: BLE001 - an old report we can't fetch just means less information
            return None

    cf = carry_forward(report, pg, day, prev)
    if cf.is_empty():
        return report
    games = dict(report.select("team", "matchup").unique().drop_nulls().iter_rows())
    meta = report.select("team", "team_name", "report_time", "game_date", "game_time").unique("team")
    cf = cf.filter(pl.col("team").is_in(list(games))).with_columns(
        pl.col("team").replace_strict(games, default=None).alias("matchup")).join(meta, on="team", how="left")
    return pl.concat([report, cf.select(report.columns)], how="vertical_relaxed")


def p_sits(row: dict) -> float:
    """P(sits) for a report row: its own estimate if we made it, else the status calibration."""
    return row["p_sit"] if row.get("p_sit") is not None else P_OUT.get(row["status"], 0.0)


CONTEXT: dict = {}  # set by gather(): data needed to line history up with tonight's roster
ROTATION_MPG = 20.0  # teammates whose absence counts as a different lineup (15 caught bench wings)
LUCK_MIN_CHANCES_PG = {"ast": 6.0, "reb": 8.0}  # volume needed before conversion luck means much
RECENT: dict = {}  # player name -> last games with potentials and a same-lineup flag
RECENT_N = 2
RECENT_MIN = {"ast": 7.0, "reb": 12.0}  # chances per same-lineup game to bother flagging
RECENT_SHORT = {"ast": 1.5, "reb": 2.0}  # expected minus actual, per game


def recent_games(pg: pl.DataFrame, player_id: int, team: str, rot: pl.DataFrame, out_keys: set,
                 n: int = RECENT_N) -> list[dict]:
    """Last n games this season with potentials, and whether the rotation around him matched
    tonight's (same rotation teammates missing; teammates who have since left are ignored)."""
    trk = (_tracking_now().filter(pl.col("player_id") == player_id)
           .drop_nulls(["potential_ast", "reb_chances"]).sort("game_date").tail(n))
    if trk.is_empty():
        return []
    mates = rot.filter(pl.col("team_now") == team, pl.col("player_id") != player_id)
    out_tonight = set(mates.filter(pl.col("player_key").is_in(list(out_keys)))["player_name"])
    out = []
    for g in trk.iter_rows(named=True):
        played = set(pg.filter(pl.col("game_id") == g["game_id"], pl.col("minutes") > 0)["player_id"])
        missing = set(mates.filter(~pl.col("player_id").is_in(list(played)))["player_name"])
        out.append({**g, "same": missing == out_tonight, "missing": missing - out_tonight,
                    "back": out_tonight - missing})
    return out[::-1]  # most recent first


def _tracking_now() -> pl.DataFrame:
    if "trk_now" not in CONTEXT:
        CONTEXT["trk_now"] = load_tracking().filter(pl.col("season_id") == "22026")
    return CONTEXT["trk_now"]


def _career_conv(player_id: int, stat: str) -> float:
    from nba_ladders.luck import CHANCES, K_CONV, league_conversion

    if "trk_hist" not in CONTEXT:
        h = load_tracking().filter(pl.col("season_id") != "22026")
        CONTEXT["trk_hist"] = (h, league_conversion(h))
    h, lg = CONTEXT["trk_hist"]
    me = h.filter(pl.col("player_id") == player_id)
    c = CHANCES[stat]
    return float((me[stat].sum() + K_CONV[stat] * lg[stat]) / (me[c].sum() + K_CONV[stat]))


POT_MISS_MIN, POT_MISS_SHORT = 12, 4.0  # last game 12+ potentials, 4+ assists short


def pot_miss(g: dict, player_id: int) -> dict | None:
    """Last game: monster potential assists that didn't convert, with tonight's lineup."""
    if not g["same"] or g["potential_ast"] < POT_MISS_MIN:
        return None
    exp = g["potential_ast"] * _career_conv(player_id, "ast")
    short = exp - g["ast"]
    if short < POT_MISS_SHORT:
        return None
    return {"short": short, "detail": f"last game {g['potential_ast']} potential assists vs {g['opponent']} "
                                      f"({g['game_date']:%d %b}), {g['ast']} assists vs {exp:.1f} expected; same lineup tonight"}


POT_VOL_MIN, POT_VOL_SPIKE = 18, 1.5  # last game 18+ potentials, 1.5x his average


def pot_baseline(player_id: int, before) -> float | None:
    """Potential assists per game before `before`: this season's last 10 (3+ games), else last season's."""
    cur = _tracking_now().filter(pl.col("player_id") == player_id, pl.col("game_date") < before) \
        .drop_nulls("potential_ast").sort("game_date").tail(10)
    if cur.height >= 3:
        return float(cur["potential_ast"].mean())
    if "trk_hist" not in CONTEXT:
        _career_conv(player_id, "ast")
    last = CONTEXT["trk_hist"][0].filter(pl.col("player_id") == player_id, pl.col("season_id") == "22025") \
        .sort("game_date").tail(10)
    return float(last["potential_ast"].mean()) if last.height >= 5 else None


def pot_volume(g: dict, player_id: int) -> dict | None:
    """Last game: a potentials spike with tonight's lineup, labelled by whether he converted."""
    if not g["same"] or g["potential_ast"] < POT_VOL_MIN:
        return None
    base = pot_baseline(player_id, g["game_date"])
    if base is None or g["potential_ast"] < POT_VOL_SPIKE * base:
        return None
    exp = g["potential_ast"] * _career_conv(player_id, "ast")
    short = exp - g["ast"]
    side = "didn't convert" if short > 0 else "converted"
    return {"short": short, "detail": f"last game {g['potential_ast']} potential assists vs {g['opponent']} "
                                      f"({g['game_date']:%d %b}), {g['potential_ast'] / base:.1f}x his {base:.1f} "
                                      f"average; {g['ast']} assists vs {exp:.1f} expected, {side}; same lineup tonight"}


def recent_signal(games: list[dict], stat: str, player_id: int) -> dict | None:
    """Same-lineup recent games where output lagged the chances. luck = blended / actual,
    on the same scale as the season luck multiplier."""
    from nba_ladders.luck import BLEND, CHANCES

    same = [g for g in games if g["same"]]
    if not same:
        return None
    ch = sum(g[CHANCES[stat]] for g in same)
    act = sum(g[stat] for g in same)
    if ch / len(same) < RECENT_MIN[stat]:
        return None
    exp = ch * _career_conv(player_id, stat)
    if (exp - act) / len(same) < RECENT_SHORT[stat]:
        return None
    luck = min((BLEND[stat] * exp + (1 - BLEND[stat]) * act) / max(act, 1), 1.6)
    label = "potential ast" if stat == "ast" else "reb chances"
    per = "; ".join(f"{g[stat]} from {g[CHANCES[stat]]} {label} vs {g['opponent']}" for g in same)
    return {"luck": luck, "detail": f"last {len(same)} same-lineup game{'s' * (len(same) > 1)}: {per} "
                                    f"(expected {exp:.1f}, got {act})"}


def recent_line(name: str, stat: str | None = None) -> str | None:
    """Per-game potentials for the last games, with the lineup flag (stat: just ast or reb)."""
    games = RECENT.get(name)
    if not games:
        return None
    parts = []
    for g in games:
        lineup = "same lineup ✓" if g["same"] else "lineup differed: " + ", ".join(
            [f"{m} out" for m in sorted(g["missing"])] + [f"{m} played" for m in sorted(g["back"])])
        a = f"{g['ast']} ast from {g['potential_ast']} potential"
        r = f"{g['reb']} reb from {g['reb_chances']} chances"
        what = {"ast": a, "reb": r}.get(stat, f"{a}, {r}")
        parts.append(f"{g['game_date']:%d %b} vs {g['opponent']}: {g['minutes']:.0f} min, {what} ({lineup})")
    return "; ".join(parts)


def _history_adjuster(team: str):
    """Back out past absences of players on `team` tonight (current roster)."""
    ctx = CONTEXT
    roster_ids = set(ctx["roster"].filter(pl.col("team_now") == team)["player_id"].to_list())

    def f(hist: pl.DataFrame) -> pl.DataFrame:
        adj = history_adjustment(ctx["pg"], hist, roster_ids, ctx["eff"], ctx["pri"], ctx["pos"])
        return adjust_history(hist, adj)
    return f


def tracking_profile(player_id: int) -> dict:
    """Per-game potential assists / rebound chances: this season (season totals) and last
    season's per-game mean and sd (variance)."""
    from nba_ladders.sources.tracking import season_totals

    out = {}
    cur = season_totals("2026-27").filter(pl.col("player_id") == player_id)
    if cur.height:
        r = cur.row(0, named=True)
        out["now"] = {"pot_ast_pg": (r["potential_ast"] or 0) / r["gp"], "reb_ch_pg": (r["reb_chances"] or 0) / r["gp"]}
    trk = load_tracking().filter(pl.col("player_id") == player_id, pl.col("season_id") == "22025")
    if trk.height >= 10:
        out["last"] = {"pot_ast_pg": trk["potential_ast"].mean(), "pot_ast_sd": trk["potential_ast"].std(),
                       "reb_ch_pg": trk["reb_chances"].mean(), "reb_ch_sd": trk["reb_chances"].std(),
                       "games": trk.height}
    return out


PRICE_RANGE = (3.0, 10.0)  # where the bets live
VALUE_EV = 0.10  # model EV at the best AU price to call a rung value


def ladder(odds: pl.DataFrame, name: str, stats: dict, focus: list[str]) -> pl.DataFrame:
    """AU rungs for the focus stats: best AU price, Sportsbet, model fair and EV at the best price."""
    k = player_key(name)
    o = odds.filter(pl.col("side") == "Over", pl.col("threshold").is_not_null(), pl.col("stat").is_in(focus),
                    pl.col("player").map_elements(player_key, return_dtype=pl.String) == k)
    if o.is_empty():
        return pl.DataFrame()
    au = o.filter(pl.col("bookmaker").is_in(AU_BOOKS))
    best = au.sort("price").group_by("stat", "threshold").last().select(
        "stat", "threshold", pl.col("price").alias("best_au"), pl.col("bookmaker").alias("best_book"))
    sb = au.filter(pl.col("bookmaker") == AU_BASE).select("stat", "threshold", pl.col("price").alias("sportsbet"))
    rows = best.join(sb, on=["stat", "threshold"], how="left")
    rows = rows.with_columns(pl.struct("stat", "threshold").map_elements(
        lambda s: model_p(stats, s["stat"], s["threshold"]), return_dtype=pl.Float64).alias("p"))
    rows = rows.filter(pl.col("p") > 0.01, pl.col("best_au").is_between(1.4, PRICE_RANGE[1] * 1.5)).with_columns(
        (1 / pl.col("p")).alias("fair"), (pl.col("best_au") * pl.col("p") - 1).alias("ev"),
        pl.col("best_au").is_between(*PRICE_RANGE).alias("in_range"))
    rows = rows.with_columns(
        pl.when(pl.col("ev") >= 1.0).then(pl.lit("big gap: check news/role"))
        .when(pl.col("in_range") & (pl.col("ev") >= VALUE_EV)).then(pl.lit("**value**"))
        .when(pl.col("ev") >= VALUE_EV).then(pl.lit("value (short price)"))
        .when(pl.col("ev") >= 0).then(pl.lit("marginal")).otherwise(pl.lit("")).alias("note"))
    # readable: in-range rungs first, then by EV; max 5 per stat
    return (rows.sort(["stat", "in_range", "ev"], descending=[False, True, True])
            .group_by("stat", maintain_order=True).head(5).sort("stat", "threshold"))


ANGLE_LABEL = {"pot_miss_ast": "monster potentials, missed", "pot_volume_ast": "potentials volume spike", "luck_ast": "season assist potentials", "starter_out": "starting centre out",
               "q_watch": "questionable teammate", "unlocked": "minutes unlocked",
               "luck_reb": "rebound chances", "foul_spot": "foul spot", "hi_var_backup": "high-variance backup",
               "model_scan": "model vs market"}
STAR_PTS = 20.0  # last season's ppg (20+ games): established stars stay in the game sections


def _is_star(player_id: int | None) -> bool:
    """Established star = 20+ ppg last season. Rookies and breakouts count as second tier."""
    if player_id is None:
        return False
    if "stars" not in CONTEXT:
        last = _frame("player_games").filter(pl.col("season_id") == "22025").group_by("player_id").agg(
            pl.col("pts").mean().alias("ppg"), pl.len().alias("gp"))
        CONTEXT["stars"] = set(last.filter(pl.col("ppg") >= STAR_PTS, pl.col("gp") >= 20)["player_id"])
    return player_id in CONTEXT["stars"]


ORDER = list(ANGLE_LABEL)


def _simulate(name: str, c: dict):
    sc = c["scenario"]
    # a q_watch candidate prices his Questionable teammate explicitly, so start from certain absences only
    src = "tonight_det" if "q_minutes" in sc else "tonight"
    base_min, base_mult = CONTEXT.get(src, {}).get(name, (0.0, {}))
    if "p_out" in sc:  # NEW starting-centre absence
        variants = [({"starter_out": True}, sc["p_out"]), ({"starter_out": False}, 1 - sc["p_out"])]
    elif "q_minutes" in sc:  # a Questionable teammate: sits (53%) or plays
        pq = sc.get("q_p", P_OUT["Questionable"])
        variants = [({"extra": sc["q_minutes"], "mult": sc["q_mult"]}, pq), ({}, 1 - pq)]
    else:
        variants = [({}, 1.0)]
    sims, by_variant = [], {}
    for v, w in variants:
        if w <= 0:
            continue
        mult = {s: base_mult.get(s, 1.0) * v.get("mult", {}).get(s, 1.0) for s in RATES}
        r = simulate_player(name, starter_out=v.get("starter_out", False), opponent=c["opp"],
                            verbose=False, n=40_000, extra_minutes=base_min + v.get("extra", 0.0),
                            rate_mult=mult, history_adjust=_history_adjuster(c["team"]))
        if r:
            sims.append((r, w))
            by_variant["sits" if (v.get("starter_out") or v.get("extra")) else "plays"] = r[0]
    if not sims:
        return None, None, by_variant
    n = 40_000
    stats = {k: np.concatenate([s[0][k][: int(n * w)] for s, w in sims]) for k in sims[0][0][0]}
    stats = apply_blowout(stats, CONTEXT.get("spreads", {}).get(c["team"]), CONTEXT.get("min20", {}).get(name))
    return stats, sims[0][0][1], by_variant


def apply_blowout(stats: dict, spread: float | None, min20: float | None, seed: int = 7) -> dict:
    """Draw a final margin per simulated game; move minutes by the blowout curve for his role
    (relative to a typical game) and scale counts with them (thinning, or extra at his rate)."""
    from nba_ladders.blowout import CURVE, MID, SD_MARGIN, TYPICAL_SPREAD, role

    if spread is None or min20 is None:
        return stats
    rng = np.random.default_rng(seed)
    n = len(stats["minutes"])
    curve = CURVE[role(min20)]
    base = np.interp(np.abs(rng.normal(TYPICAL_SPREAD, SD_MARGIN, 20_000)), MID, curve).mean()
    delta = np.interp(np.abs(rng.normal(abs(spread), SD_MARGIN, n)), MID, curve) - base
    m = np.clip(stats["minutes"], 1, None)
    f = np.clip((m + delta) / m, 0, None)
    out = {"minutes": np.clip(stats["minutes"] + delta, 0, None)}
    for s in ("pts", "reb", "ast"):
        x = stats[s]
        down = rng.binomial(x, np.minimum(f, 1.0))
        up = rng.poisson(x / m * np.clip(delta, 0, None))
        out[s] = np.where(f < 1, down, x + up)
    return out


def candidate_block(name: str, c: dict, odds: pl.DataFrame) -> tuple[list[str], list[dict]]:
    """Markdown for one candidate and its value rungs (for the shortlist)."""
    lines = [f"### {name} ({c['team']})"]
    for a, detail in c["reasons"].items():
        lines.append(f"- **{ANGLE_LABEL[a]}**: {detail}.")
    own = CONTEXT.get("avail", {}).get(player_key(name))
    if own and own[0] in ("Questionable", "Doubtful", "Out"):
        lines.append(f"- **His own status**: {own[0]}, P(plays) {1 - own[1]:.0%} ({own[2]}). Props void if he doesn't play.")
    stats, info, by_variant = _simulate(name, c)
    if stats is None:
        return lines + ["- Too little history to price."], []
    tp = tracking_profile(info["player_id"])
    trk = []
    if "now" in tp:
        trk.append(f"this season {tp['now']['pot_ast_pg']:.1f} potential ast/g, {tp['now']['reb_ch_pg']:.1f} reb chances/g")
    if "last" in tp:
        L = tp["last"]
        trk.append(f"last season {L['pot_ast_pg']:.1f} ± {L['pot_ast_sd']:.1f} potential ast, "
                   f"{L['reb_ch_pg']:.1f} ± {L['reb_ch_sd']:.1f} reb chances per game")
    lines.append(f"- **Predicted**: {stats['minutes'].mean():.0f} min (28+ min {np.mean(stats['minutes'] >= 28):.0%}); "
                 f"{stats['pts'].mean():.1f} pts / {stats['reb'].mean():.1f} reb / {stats['ast'].mean():.1f} ast. "
                 f"Adjustments: potentials AST ×{info['luck_ast']:.2f}, rebound chances ×{info['luck_reb']:.2f}; "
                 f"vs {c['opp']}: pace ×{info['opp_pace']:.2f}, points ×{info['opp_pts']:.2f}, "
                 f"assists ×{info['opp_ast']:.2f}, rebounds ×{info['opp_reb']:.2f}.")
    if trk:
        lines.append(f"- **Tracking**: {'; '.join(trk)}.")
    wl = _weakness(info["player_id"], c["opp"])
    if wl:
        lines.append(f"- **Matchup**: {wl} (what they allow per minute vs each player's own rate; in the projection).")
    c["matchup"] = wl
    c["opp_f"] = _opp_factors(info["player_id"], c["opp"])
    rl = recent_line(name)
    if rl:
        lines.append(f"- **Last {RECENT_N} games**: {rl}.")
    focus = sorted({f for a in c["reasons"] for f in FOCUS[a]})
    lad = ladder(odds, name, stats, focus)
    both = len(by_variant) == 2
    value = []
    if lad.height:
        extra_h = " | Model if he sits | Model if he plays" if both else ""
        lines += ["", f"| Rung | Sportsbet | Best AU | **Model fair** | EV @ best | Note{extra_h} |",
                  "|---|---|---|---|---|---|" + ("---|---|" if both else "")]
        for r in lad.iter_rows(named=True):
            sbp = f"${r['sportsbet']:.2f}" if r["sportsbet"] else "n/a"
            tail = ""
            if both:
                ps = model_p(by_variant["sits"], r["stat"], r["threshold"])
                pp = model_p(by_variant["plays"], r["stat"], r["threshold"])
                tail = f" | ${1 / ps:.2f} | ${1 / pp:.2f}" if ps and pp else " | | "
            lines.append(f"| {r['threshold']}+ {r['stat']} | {sbp} | ${r['best_au']:.2f} ({r['best_book']}) | "
                         f"**${r['fair']:.2f}** | {r['ev']:+.0%} | {r['note']}{tail} |")
            if r["in_range"] and r["ev"] >= VALUE_EV:
                first = next(iter(c["reasons"]))
                value.append({"player": name, "team": c["team"], "opp": c["opp"],
                              "rung": f"{r['threshold']}+ {r['stat']}",
                              "price": r["best_au"], "book": r["best_book"], "fair": r["fair"],
                              "ev": r["ev"], "angles": list(c["reasons"]),
                              "pred": (f"{stats['minutes'].mean():.0f} min · {stats['pts'].mean():.1f} pts / "
                                       f"{stats['reb'].mean():.1f} reb / {stats['ast'].mean():.1f} ast"),
                              "why": c["reasons"][first]})
        lines.append("")
    else:
        lines.append("- No AU ladder prices in the snapshot for this player yet.")
    shapes = sorted({s for a in c["reasons"] for s in SHAPES[a]})
    combos = best_combos(stats, per_shape=2, shapes=shapes)
    if combos.height:
        lines.append("- **SGM combos** (worth it if the SGM builder pays at least fair): " + "; ".join(
            f"{r['combo']} fair ${r['fair']:.2f} (legs priced independently ${r['fair_indep']:.2f})"
            for r in combos.iter_rows(named=True)))
    c["leg_prices"] = leg_prices(odds, name)
    ideas = sgm_ideas(stats, c["leg_prices"])
    if ideas:
        lines.append(f"- **SGM ideas** (builder est. = {SGM_FACTOR}× product of that book's legs): " + "; ".join(
            f"{r['combo']} ({r['book']}) fair ${r['fair']:.2f} vs est ${r['est']:.2f} ({r['ev']:+.0%}; legs {r['legs']})" for r in ideas))
    c["player_id"] = info["player_id"]
    c["pick"] = _pick(name, c, stats, lad, combos)
    c["sgm_ideas"] = ideas
    c["proj_min"] = float(stats["minutes"].mean())
    c["value_rungs"] = lad.filter(pl.col("in_range")).to_dicts() if lad.height else []
    return lines, value


SGM_FACTOR = 0.9  # builder price ≈ this x product of the book's own leg prices (user's slips: 0.87-1.2x)
SGM_RANGE = (4.0, 15.0)
LEG_MIN_P = 0.40  # each leg should be live on its own (the user's shape: low rungs, combined)
SGM_BOOKS = ("betr_au", "sportsbet")  # the user's builders; legs come from that book's own ladder
LEG_MIN = {"pts": 8, "reb": 3, "ast": 3}  # no 2r / 2a / tiny points legs


def leg_prices(odds: pl.DataFrame, name: str) -> dict:
    """{book: {(stat, threshold): price}} for the user's SGM books, single-stat rungs only."""
    k = player_key(name)
    o = odds.filter(pl.col("side") == "Over", pl.col("threshold").is_not_null(), pl.col("stat").is_in(list(LEG_MIN)),
                    pl.col("bookmaker").is_in(list(SGM_BOOKS)),
                    pl.col("player").map_elements(player_key, return_dtype=pl.String) == k)
    out = {}
    for bk, st, th, pr in o.select("bookmaker", "stat", "threshold", "price").iter_rows():
        if th >= LEG_MIN[st]:
            out.setdefault(bk, {})[(st, th)] = pr
    return out


def sgm_ideas(stats: dict, prices: dict, top: int = 3) -> list[dict]:
    """2-3 leg P/R/A combos from one book's ladder whose model fair beats the estimated builder price."""
    from itertools import combinations, product

    out = []
    for book in SGM_BOOKS:
        bp = prices.get(book, {})
        legs = {s: [k for (st, k) in bp if st == s and LEG_MIN_P <= float(np.mean(stats[s] >= k)) <= 0.85]
                for s in LEG_MIN}
        for n_legs in (2, 3):
            for ss in combinations(("pts", "reb", "ast"), n_legs):
                for ks in product(*(legs[s] for s in ss)):
                    hit = np.ones(len(stats["pts"]), dtype=bool)
                    for s, k in zip(ss, ks):
                        hit &= stats[s] >= k
                    p = float(hit.mean())
                    if p <= 0:
                        continue
                    est = SGM_FACTOR * float(np.prod([bp[(s, k)] for s, k in zip(ss, ks)]))
                    if SGM_RANGE[0] <= est <= SGM_RANGE[1]:
                        out.append({"combo": "/".join(f"{k}{s[0]}" for s, k in zip(ss, ks)), "fair": 1 / p, "est": est,
                                    "ev": est * p - 1, "book": book,
                                    "legs": " × ".join(f"${bp[(s, k)]:.2f}" for s, k in zip(ss, ks))})
    out.sort(key=lambda r: -r["ev"])
    seen, best = set(), []
    for r in out:  # one per shape
        shape = "".join(ch for ch in r["combo"] if ch.isalpha())
        if shape not in seen:
            seen.add(shape)
            best.append(r)
    return best[:top]


def _pick(name: str, c: dict, stats: dict, lad: pl.DataFrame, combos: pl.DataFrame) -> dict | None:
    """The bet: the best-EV rung priced $3-$15 in his angles' stats, plus that stat's ladder."""
    primary = min(c["reasons"], key=ORDER.index)
    best, rungs = None, []
    if lad.height:
        rng = lad.filter(pl.col("in_range")).sort("ev", descending=True)
        if rng.height:
            best = rng.row(0, named=True)
            rungs = lad.filter(pl.col("in_range"), pl.col("stat") == best["stat"]).sort("threshold").to_dicts()
    shape = "r/a" if best and best["stat"] in ("reb",) else None
    sgm = combos.filter(pl.col("shape") == (shape or SHAPES[primary][0])) if combos.height else combos
    main_stat = best["stat"] if best else FOCUS[primary][0]
    return {"player": name, "team": c["team"], "opp": c["opp"], "primary": primary,
            "angles": sorted(c["reasons"], key=ORDER.index), "why": c["reasons"][primary],
            "best": best, "rungs": rungs, "sgm": sgm.row(0, named=True) if sgm.height else None,
            "star": _is_star(c.get("player_id")), "recent": recent_line(name, "reb" if "reb" in main_stat else "ast"),
            "pred": (f"{stats['minutes'].mean():.0f} min · {stats['pts'].mean():.1f} pts / "
                     f"{stats['reb'].mean():.1f} reb / {stats['ast'].mean():.1f} ast")}


TIP_SGM_MARGIN = 1.3  # builder estimate must be >= this x model fair (room for model error)
TIP_SINGLE_MARGIN = 1.2
TIP_MIN_RISK = 1.25  # projected minutes above this x his season average -> small stake
TIP_ANGLE_WEIGHT = {"unlocked": 1.3, "starter_out": 1.3, "q_watch": 1.3, "pot_miss_ast": 1.2, "pot_volume_ast": 1.2,
                    "luck_ast": 1.1, "luck_reb": 1.1, "foul_spot": 1.1, "hi_var_backup": 1.0, "model_scan": 1.0}
TIP_N_SGM, TIP_N_SINGLE = 4, 2


def tips(cands: dict, report: pl.DataFrame, n_sgm: int = TIP_N_SGM, n_single: int = TIP_N_SINGLE,
         mixed: int | None = None) -> list[dict]:
    """The card: best SGM per player (one book's ladder, builder >= 1.3x fair), then points-first singles."""
    shaky = set(report.filter(pl.col("status").is_in(["Questionable", "Doubtful", "Out"]))["player_key"].drop_nulls())
    unfiled = set(report.filter(pl.col("status") == "Not submitted")["team"].drop_nulls())
    pg = _frame("player_games").filter(pl.col("season_id") == "22026")
    mpg = dict(pg.group_by("player_name").agg(pl.col("minutes").mean()).iter_rows())
    team_last = dict(pg.group_by("team").agg(pl.col("game_date").max()).iter_rows())
    my_last = dict(pg.group_by("player_name", "team").agg(pl.col("game_date").max()).select("player_name", "game_date").iter_rows())
    out = []
    for name, c in cands.items():
        pk = c.get("pick")
        if not pk or pk["star"] or player_key(name) in shaky:
            continue
        if c["team"] in unfiled and my_last.get(name) != team_last.get(c["team"]):
            continue  # missed his team's last game and no report yet: status unknown
        w = max(TIP_ANGLE_WEIGHT.get(a, 1.0) for a in c["reasons"])
        season = mpg.get(name)
        risk = bool(season and c.get("proj_min", 0) > TIP_MIN_RISK * season)
        why = pk["why"] + (f"; {c['matchup']}" if c.get("matchup") else "")
        base = {"player": name, "team": c["team"], "opp": c["opp"], "why": why, "risk": risk,
                "wait": c["team"] in unfiled,
                "angles": ", ".join(ANGLE_LABEL[a] for a in pk["angles"]), "proj_min": c.get("proj_min"), "season_min": season}
        for r in c.get("sgm_ideas", []):
            m = r["est"] / r["fair"]
            if TIP_SGM_MARGIN <= m < 2.0:  # beyond 2x, the market probably knows something
                out.append({**base, "kind": "SGM", "bet": r["combo"], "book": r["book"], "fair": r["fair"],
                            "price": r["est"], "take_at": TIP_SGM_MARGIN * r["fair"],
                            "score": (m - 1) * w * (0.7 if risk else 1) * _matchup_tilt(c, r["combo"])})
        for r in c.get("value_rungs", []):  # the user's books only: Sportsbet or Betr's price for the rung
            mine = [(bk, px) for bk, px in ((AU_BASE, r["sportsbet"]),
                    ("betr_au", c.get("leg_prices", {}).get("betr_au", {}).get((r["stat"], r["threshold"])))) if px]
            if not mine:
                continue
            book, price = max(mine, key=lambda x: x[1])
            r = {**r, "best_au": price, "best_book": book}
            m = r["best_au"] / r["fair"]
            if m >= TIP_SINGLE_MARGIN and m < 2.0 and PRICE_RANGE[0] <= price <= PRICE_RANGE[1]:  # EV >= 100% is usually the model, not the book
                pts_bonus = 1.15 if r["stat"] == "pts" else 1.0
                out.append({**base, "kind": "single", "bet": f"{r['threshold']}+ {r['stat']}", "book": r["best_book"],
                            "fair": r["fair"], "price": r["best_au"], "take_at": TIP_SINGLE_MARGIN * r["fair"],
                            "score": (m - 1) * w * pts_bonus * (0.7 if risk else 1) * _matchup_tilt(c, r["stat"])})
    picks, used = [], set()
    if mixed:  # top N of any kind, one per player
        for r in sorted(out, key=lambda x: -x["score"]):
            if len(picks) >= mixed:
                break
            if r["player"] not in used:
                picks.append(r)
                used.add(r["player"])
        return picks
    for kind, n in (("SGM", n_sgm), ("single", n_single)):
        for r in sorted((x for x in out if x["kind"] == kind), key=lambda x: -x["score"]):
            if sum(p["kind"] == kind for p in picks) >= n:
                break
            if r["player"] not in used:
                picks.append(r)
                used.add(r["player"])
    return picks


def _weakness(player_id: int, opp: str) -> str:
    from nba_ladders.absences import position_groups
    from nba_ladders.matchup import weakness_line

    if "grp" not in CONTEXT:
        CONTEXT["grp"] = dict(position_groups(positions()).iter_rows())
    g = CONTEXT["grp"].get(player_id)
    return weakness_line(opp, g, CONTEXT.get("day")) if g and CONTEXT.get("day") else ""


def _opp_factors(player_id: int, opp: str) -> dict:
    from nba_ladders.matchup import group_factors

    g = CONTEXT.get("grp", {}).get(player_id)
    if not g or not CONTEXT.get("day"):
        return {}
    r = group_factors(CONTEXT["day"]).filter(pl.col("opponent") == opp, pl.col("grp") == g)
    return {s: r[s][0] for s in ("pts", "reb", "ast")} if r.height else {}


TIP_MATCHUP_TILT = 2.0  # ranking: +5% allowed in the bet's stats -> +10% score


def _matchup_tilt(c: dict, bet: str) -> float:
    f = c.get("opp_f") or {}
    stats = re.findall(r"(pts|reb|ast)", bet) or [LEG[x] for x in re.findall(r"\d+([pra])", bet)]
    vals = [f[s] for s in stats if s in f]
    return 1 + TIP_MATCHUP_TILT * (float(np.mean(vals)) - 1) if vals else 1.0


LEG = {"p": "pts", "r": "reb", "a": "ast"}


def _injuries(report: pl.DataFrame, team: str) -> str:
    r = report.filter(pl.col("team") == team, pl.col("status").is_in(["Out", "Doubtful", "Questionable"]),
                      ~pl.col("reason").str.contains("(?i)g ?league|two-way"))
    if r.is_empty():
        return "nobody significant listed"
    return "; ".join(f"{x['player']} {x['status']}" + (f" ({x['reason']})" if x["reason"].startswith("B2B") else "")
                     for x in r.iter_rows(named=True))


def render(day: date, cands: dict, report: pl.DataFrame, qwatch: dict | None = None) -> str:
    qwatch = qwatch or {}
    odds = latest(load_cached("live"))
    lo = datetime(day.year, day.month, day.day, 6, tzinfo=ET)
    hi = lo + timedelta(days=1)
    odds = odds.filter(pl.col("commence_time").map_elements(lambda s: lo <= datetime.fromisoformat(s) < hi,
                                                            return_dtype=pl.Boolean))
    not_filed = report.filter(pl.col("status") == "Not submitted")["team"].drop_nulls().to_list()
    snap = odds["snapshot"].max() if odds.height else "none (run odds_api snapshot)"
    games = report.select("matchup").unique().drop_nulls()["matchup"].sort().to_list()
    blocks, shortlist = {}, []
    for name, c in sorted(cands.items(), key=lambda kv: min(ORDER.index(a) for a in kv[1]["reasons"])):
        b, v = candidate_block(name, c, odds)
        if set(c["reasons"]) == {"model_scan"} and not v:  # scanned, no value: leave it out
            c.pop("pick", None)
            continue
        game = next(g for g in games if c["team"] in g.split("@"))
        blocks.setdefault(game, []).append((c, b))
        shortlist += v
    lines = [f"# Game-day write-up: {day} (US Eastern)", "",
             (f"Injury report {report['report_time'].max()} ET · odds snapshot {snap} (Sportsbet, TAB, Dabble, "
              "Betr, Neds, Ladbrokes, PointsBet; US books for reference).")]
    if not_filed:
        lines.append(f"\n**Not yet submitted:** {', '.join(sorted(not_filed))}. Re-run closer to tip.")
    lines += ["", ("**How to read this**: *Model fair* = the projection alone: minutes as a scenario mix, per-minute "
                   "rates lined up with tonight's lineup (absences in and out), opponent pace and defence (points, "
                   "assists), opponent rebounding, and potentials / rebound chances. EV = best AU price × model "
                   f"probability − 1; **value** = EV ≥ {VALUE_EV:.0%} on a rung priced ${PRICE_RANGE[0]:.0f}–"
                   f"${PRICE_RANGE[1]:.0f}. No backtest adjustments. SGM fair prices are model-only: compare with the builder.")]
    picks = [c["pick"] for c in cands.values() if c.get("pick") and c["pick"]["best"]]
    stars = sorted(p["player"] for p in picks if p["star"])
    picks = sorted((p for p in picks if not p["star"]), key=lambda p: -p["best"]["ev"])
    picks = [p for p in picks if p["angles"] != ["model_scan"] or p["best"]["ev"] >= VALUE_EV]

    def rung(r):
        return f"{r['threshold']}+ ${r['best_au']:.2f} (fair ${r['fair']:.2f})"

    def card_rows(card):
        rows = ["", "| # | Bet | Book | Model fair | Price | Take at | Stake | Why |", "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(card, 1):
            est = "est. " if r["kind"] == "SGM" else ""
            mins = (f"; {r['proj_min']:.0f} min vs {r['season_min']:.0f} season" if r["season_min"] else "")
            rows.append(f"| {i} | **{r['player']}** ({r['team']} vs {r['opp']}) {r['kind']} **{r['bet']}** | {r['book']} | "
                        f"${r['fair']:.2f} | {est}${r['price']:.2f} | **${r['take_at']:.2f}** | "
                        f"{'small' if r['risk'] else 'standard'}{' · **wait for report**' if r['wait'] else ''} | "
                        f"{r['angles']}: {r['why']}{mins} |")
        return rows

    top10 = tips(cands, report, mixed=10)
    lines += ["", "## Top 10", "",
              ("The ten best bets on the slate, SGMs and singles together, one per player, ranked by margin over model fair "
               "weighted to your angles. Same filters as the tips: your books, available players only, builder ≥ "
               f"{TIP_SGM_MARGIN}× fair for SGMs, singles $3–$10 at ≥ {TIP_SINGLE_MARGIN}× fair.")]
    lines += card_rows(top10) if top10 else ["", "Nothing clears the margins."]
    card = tips(cands, report)
    lines += ["", "## My tips", "",
              (f"SGMs from one book's own ladder where the builder estimate is at least {TIP_SGM_MARGIN}× model fair, then "
               f"singles at {TIP_SINGLE_MARGIN}× fair or better (points first). Ranked by margin, weighted to your angles "
               "(minutes unlocked, starting centre out, Questionable teammate, potentials). Players who are Questionable "
               f"themselves are left out. **Small** = model minutes more than {TIP_MIN_RISK}× his season average. "
               "Take each only at the 'take at' price or better.")]
    if card:
        lines += card_rows(card)
    else:
        lines.append("\nNo tips clear the margins at current prices.")
    lines += ["", f"## Bets: model vs market, ${PRICE_RANGE[0]:.0f}–${PRICE_RANGE[1]:.0f}", "",
              ("Second-tier players (stars = 20+ ppg last season), ranked by EV of their best rung in range. "
               "The ladder is every in-range rung of that stat. "
               + (f"Stars with live angles are in the game sections: {', '.join(stars)}." if stars else ""))]
    if picks:
        lines += ["", "| # | Player | Angle | Bet (best AU) | Model fair | EV | Ladder in range | SGM idea | Predicted |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for i, s in enumerate([p for p in picks if p["best"]["ev"] > 0][:15], 1):
            b, g = s["best"], s["sgm"]
            ang = ", ".join(ANGLE_LABEL[a] for a in s["angles"])
            why = f"{ang}<br><small>{s['why']}" + (f"; {s['recent']}" if s["recent"] else "") + "</small>"
            tag = " **value**" if b["ev"] >= VALUE_EV else ""
            lines.append(f"| {i} | **{s['player']}** ({s['team']} vs {s['opp']}) | {why} | "
                         f"**{b['threshold']}+ {b['stat']} ${b['best_au']:.2f}** ({b['best_book']}){tag} | "
                         f"${b['fair']:.2f} | {b['ev']:+.0%} | {' → '.join(rung(r) for r in s['rungs'])} | "
                         f"{(g['combo'] + ' fair $' + format(g['fair'], '.2f')) if g else 'n/a'} | {s['pred']} |")
        neg = [p["player"] for p in picks if p["best"]["ev"] <= 0]
        if neg:
            lines.append(f"\nNo positive in-range rung: {', '.join(neg)}.")
    else:
        lines.append("\nNo in-range prices for tonight's candidates yet.")
    sg = [(c, r) for c in cands.values() if not (c.get("pick") or {}).get("star", True) for r in c.get("sgm_ideas", [])]
    sg = sorted(sg, key=lambda cr: -cr[1]["ev"])
    lines += ["", "## SGM ladders: low rungs combined", "",
              (f"Each leg live on its own (model {LEG_MIN_P:.0%}+), combined into ${SGM_RANGE[0]:.0f}–${SGM_RANGE[1]:.0f}. "
               f"Legs come from one book's own ladder (Betr: pts 8/10/12/15/18/20/25, reb 3/5/7/10/12, ast 3/5/7/10; "
               f"Sportsbet: pts in 5s, reb/ast in 2s), no 2r/2a. Builder price ≈ {SGM_FACTOR}× the product of that book's "
               "legs (your three winning slips paid 0.87–1.2×): check the builder. Second-tier players only.")]
    if sg:
        lines += ["", "| Player | Combo | Book | Model fair | Builder est. | EV | Legs |", "|---|---|---|---|---|---|---|"]
        for c, r in [x for x in sg if x[1]["ev"] > 0][:12]:
            lines.append(f"| **{c['pick']['player']}** ({c['team']} vs {c['opp']}) | **{r['combo']}** | {r['book']} | ${r['fair']:.2f} | "
                         f"${r['est']:.2f} | {r['ev']:+.0%} | {r['legs']} |")
    else:
        lines.append("\nNone priced yet.")
    lines += ["", f"## Every value rung (EV ≥ {VALUE_EV:.0%}, ${PRICE_RANGE[0]:.0f}–${PRICE_RANGE[1]:.0f}), stars included", ""]
    if shortlist:
        lines += ["| Player | Rung | Best AU | Model fair | EV | Angles |", "|---|---|---|---|---|---|"]
        for s in sorted(shortlist, key=lambda s: -s["ev"])[:20]:
            lines.append(f"| {s['player']} ({s['team']}) | {s['rung']} | ${s['price']:.2f} ({s['book']}) | "
                         f"${s['fair']:.2f} | {s['ev']:+.0%} | {', '.join(ANGLE_LABEL[a] for a in s['angles'])} |")
    else:
        lines.append("None at current prices.")
    for g in games:
        away, home = g.split("@")
        lines += ["", f"## {away} @ {home}", "",
                  f"- **{away}**: {_injuries(report, away)}", f"- **{home}**: {_injuries(report, home)}"]
        for team in (away, home):
            for w in qwatch.get(team, []):
                lines.append(f"- **Questionable watch ({team})**: {w}. _Model gains are shrunk; historically a "
                             "starting centre sitting gives his backup ~+5 min._")
        items = blocks.get(g, [])
        if not items:
            lines.append("- No candidates from any angle.")
            continue
        angles = sorted({a for c, _ in items for a in c["reasons"]}, key=ORDER.index)
        lines.append(f"- **Angles live**: {', '.join(ANGLE_LABEL[a] for a in angles)} "
                     f"({len(items)} candidate{'s' if len(items) != 1 else ''}).")
        for _, b in items:
            lines += [""] + b
    return "\n".join(lines) + "\n"


CSS = """
:root { --bg:#f7f7f5; --card:#ffffff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e3e3e0;
        --accent:#2f5bd3; --good:#137a3a; --goodbg:#e6f4ea; --bad:#a3302a; --badbg:#fbeaea; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#131315; --card:#1c1c1f; --ink:#ececee; --muted:#9a9aa1; --line:#2e2e33;
  --accent:#7aa2ff; --good:#5fd38a; --goodbg:#163222; --bad:#ff7b72; --badbg:#3a1a1a; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; }
h2 { font-size:19px; margin:32px 0 10px; }
h3 { font-size:16px; margin:0; display:inline; }
p, li { color:var(--ink); }
em { color:var(--muted); }
code { background:var(--line); padding:1px 5px; border-radius:4px; font-size:13px; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; margin:10px 0 14px; font-size:13.5px; background:var(--card); }
th, td { border-bottom:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; background:var(--bg); position:sticky; top:0; }
.game { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:4px 18px 14px; margin:18px 0; }
.game > h2 { margin-top:14px; }
details { border-top:1px solid var(--line); padding:10px 0 4px; }
summary { cursor:pointer; list-style:none; }
summary::-webkit-details-marker { display:none; }
summary::before { content:"▸ "; color:var(--muted); }
details[open] > summary::before { content:"▾ "; }
.ev-pos { color:var(--good); background:var(--goodbg); font-weight:600; border-radius:4px; padding:1px 5px; }
.ev-neg { color:var(--bad); }
.top5 table { font-size:14px; }
.top5 td:nth-child(2) { white-space:nowrap; }
.note { color:var(--muted); font-size:13px; }
"""


def to_html(md_text: str, title: str) -> str:
    """Markdown -> styled standalone HTML: game cards, collapsible candidates, EV highlighting."""
    import re

    import markdown

    body = markdown.markdown(md_text, extensions=["tables"])
    body = body.replace("<table>", '<div class="wrap"><table>').replace("</table>", "</table></div>")
    body = re.sub(r"<td>([+−-]\d+%)</td>",
                  lambda m: f'<td><span class="{"ev-pos" if m.group(1).startswith("+") else "ev-neg"}">'
                            f"{m.group(1)}</span></td>", body)
    # top sections: wrap "Top 5" in a highlighted block
    body = body.replace("<h2>Picks: the spots</h2>", '<section class="top5"><h2>Picks: the spots</h2>', 1)
    body = body.replace("<h2>Shortlist", "</section><h2>Shortlist", 1)
    # each game (h2 "AWAY @ HOME") becomes a card; each candidate (h3) a collapsible section
    parts = re.split(r"(?=<h2>[A-Z]{2,3} @ [A-Z]{2,3}</h2>)", body)
    out = [parts[0]]
    for g in parts[1:]:
        segs = re.split(r"(?=<h3>)", g)
        head, cands = segs[0], segs[1:]
        html = f'<section class="game">{head}'
        for c in cands:
            m = re.match(r"<h3>(.*?)</h3>(.*)", c, re.DOTALL)
            html += f"<details><summary><h3>{m.group(1)}</h3></summary>{m.group(2)}</details>" if m else c
        out.append(html + "</section>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{title}</title><style>{CSS}</style></head><body><main>{''.join(out)}</main></body></html>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, default=datetime.now(ET).date())
    args = ap.parse_args()
    cands, report, qwatch = gather(args.date)
    md = render(args.date, cands, report, qwatch)
    out = config.ROOT / "docs" / "slates" / f"{args.date.isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    html = out.with_suffix(".html")
    html.write_text(to_html(md, f"NBA slate {args.date}"))
    print(md)
    print(f"wrote {out}\nwrote {html}")


if __name__ == "__main__":
    main()
