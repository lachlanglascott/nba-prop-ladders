"""Candidates report: every angle's candidates for a game day, with evidence, prediction, prices.

    uv run python -m nba_ladders.candidates                 # today's ET slate
    uv run python -m nba_ladders.candidates --date 2026-10-30

Angles (all kept by choice; each carries what the two-season Sportsbet backtest found):
starter out (backup bigs), minutes unlocked (absences), potentials luck (AST / REB),
foul spot (backup big vs foul-prone starter + high-FTA opponent), high-variance backup bigs.

Per candidate: reasons, prediction (minutes, means, tracking volume and per-game variance),
ladder prices (Sportsbet, best AU, fair, EV), and SGM combos with fair prices for comparing
against the SGM builder. Needs a same-day odds snapshot for prices
(`python -m nba_ladders.sources.odds_api snapshot`). Writes data/processed/sheets/.
"""

import argparse
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
from nba_ladders.market import TEAM_ABBR, cushion, latest, model_p
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
FOUL_Q = 0.667  # top third of starter fouls/36 x opponent FTA, as in the backtest
HI_VAR_MIN_SD = 5.8  # top tercile of minutes sd (backtest)

EVIDENCE = {
    "starter_out": "Two-season blind tails −13% at Sportsbet; the model adds real information "
                   "here (coef 0.24), so it's about price and SGM structure, not blind betting.",
    "unlocked": "More minutes from absences is mostly priced in (−15% blind tails).",
    "luck_ast": "Best single-leg signal: assist luck ≥1.10 at best AU price +3.7% (+3.8%/+3.5% by "
                "season, CI includes 0).",
    "luck_reb": "Rebound luck: thin and mixed in the backtest; treat as a lead.",
    "foul_spot": "Faded on two seasons (−21% blind tails); a weak signal on its own.",
    "hi_var_backup": "High-variance backups: −16% blind tails over two seasons; the variance "
                     "is mostly priced. Your SGM-ladder angle.",
    "q_watch": "Questionable teammate: priced as a 53/47 mix (calibrated sit rate). If he's ruled "
               "out, the 'if he sits' column is the live price, and books may be slow to move.",
}
FOCUS = {"starter_out": ["reb", "pts+reb", "pts"], "unlocked": ["pts", "reb", "pts+reb", "ast"],
         "luck_ast": ["ast"], "luck_reb": ["reb", "pts+reb"], "foul_spot": ["reb", "pts+reb"],
         "hi_var_backup": ["reb", "pts+reb"], "q_watch": ["pts", "reb", "pts+reb", "ast"]}
SHAPES = {"starter_out": ["p/r", "p/r/a"], "unlocked": ["p/r", "p/a", "p/r/a"],
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
                f"{qr['player_name']} Questionable ({qr['reason']}); if he sits: +{r['gain']:.1f} min",
                strength=r["gain"], q_minutes=r["gain"], q_mult={s: r[f"mult_{s}"] for s in RATES})

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
        for stat, angle in (("ast", "luck_ast"), ("reb", "luck_reb")):
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

    # tonight's absences (current teammates Out/Doubtful) for EVERY player: the history
    # adjustment backs out past absences of current teammates, so this adds tonight's back
    tonight = {r["player_name"]: (r["shift_min"], {s: r[f"mult_{s}"] for s in RATES})
               for r in ctx.iter_rows(named=True)}
    CONTEXT.update(pg=pg, eff=eff, pri=pri, pos=pos, roster=roster, tonight=tonight)
    return cands, report, qwatch


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
        out["now"] = {"pot_ast_pg": r["potential_ast"] / r["gp"], "reb_ch_pg": r["reb_chances"] / r["gp"]}
    trk = load_tracking().filter(pl.col("player_id") == player_id, pl.col("season_id") == "22025")
    if trk.height >= 10:
        out["last"] = {"pot_ast_pg": trk["potential_ast"].mean(), "pot_ast_sd": trk["potential_ast"].std(),
                       "reb_ch_pg": trk["reb_chances"].mean(), "reb_ch_sd": trk["reb_chances"].std(),
                       "games": trk.height}
    return out


CAL_CUSHION = 0.05  # calibrated EV needed to shortlist (the blend already discounts the model)


def ladder(odds: pl.DataFrame, name: str, stats: dict, focus: list[str], luck_raw: dict | None = None,
           starter_out: bool = False) -> pl.DataFrame:
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
    rows = rows.filter(pl.col("p").is_between(0.06, 0.75)).with_columns(
        (1 / pl.col("p")).alias("fair"), (pl.col("best_au") * pl.col("p") - 1).alias("ev_best"))
    if rows.height:
        from nba_ladders.blend import predict

        luck_raw = luck_raw or {}
        book = rows.select(pl.coalesce("sportsbet", "best_au")).to_series().to_numpy()
        stat = rows["stat"].to_numpy()
        luck_hi = np.array([luck_raw.get("ast" if s == "ast" else "reb", 1.0) >= 1.10 for s in stat])
        p_cal = predict(1 / book, rows["p"].to_numpy(), stat, luck_hi, np.full(len(stat), starter_out))
        rows = rows.with_columns(pl.Series("p_cal", p_cal)).with_columns(
            (1 / pl.col("p_cal")).alias("fair_cal"), (pl.col("best_au") * pl.col("p_cal") - 1).alias("ev_cal"))
    rows = rows.with_columns(pl.col("p").map_elements(cushion, return_dtype=pl.Float64).alias("cush"))
    rows = rows.with_columns(
        pl.when(pl.col("ev_best") >= 1.0).then(pl.lit("model way off book: check news"))
        .when(pl.col("ev_cal") >= CAL_CUSHION).then(pl.lit("**value (calibrated)**"))
        .when(pl.col("ev_cal") >= 0).then(pl.lit("marginal")).otherwise(pl.lit("")).alias("note"))
    # keep it readable: Sportsbet's rungs first, then the best calibrated-EV rungs; max 4 per stat
    rows = rows.with_columns(pl.col("sportsbet").is_not_null().alias("sb")).sort(
        ["stat", "sb", "ev_cal"], descending=[False, True, True])
    return rows.group_by("stat", maintain_order=True).head(4).sort("stat", "threshold").drop("sb")


ANGLE_LABEL = {"luck_ast": "assist potentials", "starter_out": "starting centre out",
               "q_watch": "questionable teammate", "unlocked": "minutes unlocked",
               "luck_reb": "rebound chances", "foul_spot": "foul spot", "hi_var_backup": "high-variance backup"}
BASELINE_PRICE = (1.5, 5.0)  # the first rung sits in a realistic range; the ladder climbs from there
LADDER_MAX, LADDER_RUNGS = 15.0, 3  # rungs shown above the baseline
STAR_PTS = 20.0  # projected points: stars stay in the game sections, picks are the second tier


def spot_score(c: dict) -> float:
    """How strongly the angles fire (not EV). Ranks picks within each angle group."""
    st, out = c.get("strength", {}), []
    for a in c["reasons"]:
        v = st.get(a, 0.0)
        if a == "luck_ast":
            out.append(2.0 + 10 * (v - 1.05))
        elif a == "starter_out":
            out.append(1.0 + 2.5 * v if "p_out" in c["scenario"] else 0.3)  # ongoing absence: weak
        elif a == "q_watch":
            out.append(1.5 + 0.3 * v)
        elif a == "unlocked":
            out.append(1.2 + 0.2 * (v - 3))
        elif a == "luck_reb":
            out.append(1.0 + 5 * (v - 1.05))
        else:
            out.append(0.5 if a == "foul_spot" else 0.4)
    return max(out) + 0.2 * (len(out) - 1)
ORDER = list(ANGLE_LABEL)


def _simulate(name: str, c: dict):
    sc = c["scenario"]
    base_min, base_mult = CONTEXT.get("tonight", {}).get(name, (0.0, {}))
    if "p_out" in sc:  # NEW starting-centre absence
        variants = [({"starter_out": True}, sc["p_out"]), ({"starter_out": False}, 1 - sc["p_out"])]
    elif "q_minutes" in sc:  # a Questionable teammate: sits (53%) or plays
        pq = P_OUT["Questionable"]
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
    return stats, sims[0][0][1], by_variant


def candidate_block(name: str, c: dict, odds: pl.DataFrame) -> tuple[list[str], list[dict]]:
    """Markdown for one candidate and its value rungs (for the shortlist)."""
    lines = [f"### {name} ({c['team']})"]
    for a, detail in c["reasons"].items():
        lines.append(f"- **{ANGLE_LABEL[a]}**: {detail}. _{EVIDENCE[a]}_")
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
                 f"Adjustments: luck AST ×{info['luck_ast']:.2f}, REB ×{info['luck_reb']:.2f}; "
                 f"vs {c['opp']} rebounds ×{info['opp_reb']:.2f}.")
    if trk:
        lines.append(f"- **Tracking**: {'; '.join(trk)}.")
    rl = recent_line(name)
    if rl:
        lines.append(f"- **Last {RECENT_N} games**: {rl}.")
    focus = sorted({f for a in c["reasons"] for f in FOCUS[a]})
    from nba_ladders.luck import live_luck

    lr = live_luck().filter(pl.col("player_id") == info["player_id"])
    luck_raw = {"ast": lr["luck_ast"][0], "reb": lr["luck_reb"][0]} if lr.height else {}
    lad = ladder(odds, name, stats, focus, luck_raw, "p_out" in c["scenario"])
    both = len(by_variant) == 2
    value = []
    if lad.height:
        extra_h = " | Model if he sits | Model if he plays" if both else ""
        lines += ["", f"| Rung | Sportsbet | Best AU | Model fair | **Calibrated fair** | EV @ best (cal.) | Note{extra_h} |",
                  "|---|---|---|---|---|---|---|" + ("---|---|" if both else "")]
        for r in lad.iter_rows(named=True):
            sbp = f"${r['sportsbet']:.2f}" if r["sportsbet"] else "n/a"
            tail = ""
            if both:
                ps = model_p(by_variant["sits"], r["stat"], r["threshold"])
                pp = model_p(by_variant["plays"], r["stat"], r["threshold"])
                tail = f" | ${1 / ps:.2f} | ${1 / pp:.2f}" if ps and pp else " | | "
            lines.append(f"| {r['threshold']}+ {r['stat']} | {sbp} | ${r['best_au']:.2f} ({r['best_book']}) | "
                         f"${r['fair']:.2f} | ${r['fair_cal']:.2f} | {r['ev_cal']:+.0%} | {r['note']}{tail} |")
            if r["note"] == "**value (calibrated)**":
                first = next(iter(c["reasons"]))
                value.append({"player": name, "team": c["team"], "opp": c["opp"],
                              "rung": f"{r['threshold']}+ {r['stat']}",
                              "price": r["best_au"], "book": r["best_book"], "fair": r["fair_cal"],
                              "model_fair": r["fair"], "ev": r["ev_cal"], "angles": list(c["reasons"]),
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
    c["pick"] = _pick(name, c, stats, lad, combos)
    return lines, value


def _pick(name: str, c: dict, stats: dict, lad: pl.DataFrame, combos: pl.DataFrame) -> dict | None:
    """Baseline bet for the spot: the primary angle's main stat at a realistic price."""
    primary = min(c["reasons"], key=ORDER.index)
    main_stat = FOCUS[primary][0]
    base, ladder_up = None, []
    if lad.height:
        rows = lad.filter(pl.col("stat") == main_stat).sort("threshold")
        ok = rows.filter(pl.col("best_au").is_between(*BASELINE_PRICE))
        if ok.height:
            # the rung closest to a ~$2.50 price, preferring the better calibrated EV
            base = ok.with_columns(((pl.col("best_au") - 2.5).abs() - 2 * pl.col("ev_cal")).alias("k")) \
                .sort("k").row(0, named=True)
            up = rows.filter(pl.col("threshold") > base["threshold"], pl.col("best_au") <= LADDER_MAX)
            ladder_up = up.head(LADDER_RUNGS).to_dicts()
    sgm = combos.filter(pl.col("shape") == SHAPES[primary][0]) if combos.height else combos
    return {"player": name, "team": c["team"], "opp": c["opp"], "primary": primary,
            "angles": sorted(c["reasons"], key=ORDER.index), "score": spot_score(c),
            "why": c["reasons"][primary], "base": base, "up": ladder_up,
            "sgm": sgm.row(0, named=True) if sgm.height else None,
            "star": float(stats["pts"].mean()) >= STAR_PTS, "recent": recent_line(name, "reb" if "reb" in main_stat else "ast"),
            "pred": (f"{stats['minutes'].mean():.0f} min · {stats['pts'].mean():.1f} pts / "
                     f"{stats['reb'].mean():.1f} reb / {stats['ast'].mean():.1f} ast")}


def _injuries(report: pl.DataFrame, team: str) -> str:
    r = report.filter(pl.col("team") == team, pl.col("status").is_in(["Out", "Doubtful", "Questionable"]),
                      ~pl.col("reason").str.contains("(?i)g ?league|two-way"))
    if r.is_empty():
        return "nobody significant listed"
    return "; ".join(f"{x['player']} {x['status']}" for x in r.iter_rows(named=True))


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
        game = next(g for g in games if c["team"] in g.split("@"))
        blocks.setdefault(game, []).append((c, b))
        shortlist += v
    lines = [f"# Game-day write-up: {day} (US Eastern)", "",
             (f"Injury report {report['report_time'].max()} ET · odds snapshot {snap} (Sportsbet, TAB, Dabble, "
              "Betr, Neds, Ladbrokes, PointsBet; US books for reference).")]
    if not_filed:
        lines.append(f"\n**Not yet submitted:** {', '.join(sorted(not_filed))}. Re-run closer to tip.")
    lines += ["", ("**How to read this**: *Model fair* = the simulator alone. *Calibrated fair* = Sportsbet's price "
                   "and the model blended with weights learned on two seasons of results (out of season it beats "
                   "Sportsbet's own price: log loss 0.5602 vs 0.5667). EV uses the calibrated fair at the best AU price. "
                   "Raw model-vs-book gaps lose money historically, so trust the calibrated column. SGM combos are "
                   "model-only (no SGM prices to calibrate against): compare them with the builder.")]
    picks = [c["pick"] for c in cands.values() if c.get("pick") and c["pick"]["base"]]
    for p in picks:  # value only nudges the ranking; a clearly bad price drops the spot
        ev = p["base"]["ev_cal"]
        p["rank"] = p["score"] + (min(ev, 0.3) if ev > 0 else (-1.0 if ev < -0.05 else 0.0))

    def rung(r, bold=False):
        s = f"{r['threshold']}+ ${r['best_au']:.2f}"
        return (f"**{s}**" if bold else s) + f" (fair ${r['fair_cal']:.2f}, {r['best_book']})"

    def pick_rows(group):
        out = []
        for i, s in enumerate(sorted(group, key=lambda s: -s["rank"]), 1):
            b, g = s["base"], s["sgm"]
            ladder = " → ".join([rung(b, bold=True)] + [rung(u) for u in s["up"]])
            sgm = f"{g['combo']} fair ${g['fair']:.2f}" if g else "n/a"
            why = s["why"] + (f"<br><small>{s['recent']}</small>" if s["recent"] else "")
            out.append(f"| {i} | **{s['player']}** ({s['team']} vs {s['opp']}) | {why} | {b['stat']}: {ladder} | "
                       f"{b['ev_cal']:+.0%} | {sgm} | {s['pred']} |")
        return out

    head = [("| # | Player | The spot (last 2 games: potentials, lineup) | Ladder (best AU, calibrated fair) | "
             "EV 1st rung | SGM idea | Predicted |"), "|---|---|---|---|---|---|---|"]
    stars = sorted(p["player"] for p in picks if p["star"])
    picks = [p for p in picks if not p["star"]]
    lines += ["", "## Picks: the spots", "",
              (f"Second-tier players only (projected under {STAR_PTS:.0f} pts), chosen by angle strength, not EV. "
               f"Each ladder starts at a $1.50–$5.00 rung and climbs up to ${LADDER_MAX:.0f}. "
               "Potentials only count from recent games where the rotation matched tonight's. "
               + (f"Stars with live angles are in the game sections: {', '.join(stars)}." if stars else ""))]
    ast = [p for p in picks if p["primary"] == "luck_ast"]
    rest = [p for p in picks if p["primary"] != "luck_ast"]
    lines += ["", "### Assist potentials", ""] + (head + pick_rows(ast)[:6] if ast else ["None tonight."])
    lines += ["", "### Other angles", ""] + (head + pick_rows(rest)[:8] if rest else ["None tonight."])
    lines += ["", f"## Shortlist: calibrated EV ≥ {CAL_CUSHION:.0%} at the best AU price", ""]
    if shortlist:
        lines += ["| Player | Rung | Best AU | Calibrated fair | Model fair | Cal. EV | Angles |",
                  "|---|---|---|---|---|---|---|"]
        for s in sorted(shortlist, key=lambda s: -s["ev"])[:15]:
            lines.append(f"| {s['player']} ({s['team']}) | {s['rung']} | ${s['price']:.2f} ({s['book']}) | "
                         f"${s['fair']:.2f} | ${s['model_fair']:.2f} | {s['ev']:+.0%} | "
                         f"{', '.join(ANGLE_LABEL[a] for a in s['angles'])} |")
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
