"""Daily game-day sheet: injury report -> starting centres out -> backup bigs' fair ladders.

    uv run python -m nba_ladders.sheet                 # today's games (US Eastern date)
    uv run python -m nba_ladders.sheet --date 2026-10-24

Alerts (starting centre Out/Doubtful) come first; Questionable centres are priced as a mix
of "plays" and "sits". Writes data/processed/sheets/sheet_<date>.md.
"""

import argparse
from datetime import date, datetime, timedelta

import polars as pl
from nba_api.stats.static import teams

from nba_ladders import config
from nba_ladders.absences import (
    MIN_MATE_MINUTES,
    RATES,
    context_effects,
    game_absences,
    pair_effects,
    priors,
    shrunk_effects,
)
from nba_ladders.context_sim import fit_context, simulate_context
from nba_ladders.luck import live_luck
from nba_ladders.price import player_history
from nba_ladders.rebounding import live_multiplier_table, live_rebound_multiplier
from nba_ladders.scenarios import CENTRE_REB36, positions
from nba_ladders.screens import BIG_POSITIONS, CENTRE_POSITIONS, parse_player_index
from nba_ladders.sim import combo_probs, fit_player, parse_combo, simulate_stats
from nba_ladders.sources.injury_report import ET, player_key, report_for_date
from nba_ladders.sources.nba_stats import get_player_index

# P(doesn't play) by status on the ~5 PM ET report, calibrated on 2025-26: 13,392 listings over
# 164 game days vs box scores (python -m nba_ladders.status_calibration; docs/status_calibration.md).
# "Available" sits 24% mostly via two-way/G-League designations, so it's treated as playing here.
P_OUT = {"Out": 0.999, "Doubtful": 0.991, "Questionable": 0.53, "Probable": 0.12, "Available": 0.0}
ALERT_STATUSES = {"Out", "Doubtful"}
WATCH_STATUSES = {"Questionable"}
RECENT_GAMES = 15
ACTIVE_DAYS = 45  # no game in this window = long-term absent; not the current starter
BACKUPS_PER_TEAM = 2
COMBOS = ["8p/8r", "10p/10r", "12p/12r", "15p/12r", "10p/8r/2a", "12p/10r/3a"]
TEAM_ABBR = {t["full_name"].replace(" ", ""): t["abbreviation"] for t in teams.get_teams()}
UNLOCK_MIN_SHIFT = 3.0  # list players projected to gain at least this many minutes
UNLOCK_COMBOS = ["15p", "20p", "15p/5r", "15p/8r", "10p/8r/2a", "20p/5r/5a"]
HISTORY_GAMES = 120


def depth_chart(pg: pl.DataFrame, roster: pl.DataFrame, pos: pl.DataFrame) -> pl.DataFrame:
    """Bigs per current team, ranked by recent minutes, with the starting centre flagged.

    roster: player_id -> team_now (current player index). Recent minutes are each player's
    last RECENT_GAMES games on any team, so offseason moves are handled. Players who haven't
    played in the last ACTIVE_DAYS (long-term injured) rank below everyone active, so the
    depth chart reflects who is actually starting now.
    """
    cutoff = pg["game_date"].max() - timedelta(days=ACTIVE_DAYS)
    recent = (
        pg.sort("game_date")
        .group_by("player_id")
        .tail(RECENT_GAMES)
        .group_by("player_id")
        .agg(
            pl.col("player_name").last(),
            pl.col("minutes").mean().alias("recent_min"),
            (pl.col("reb").sum() / pl.col("minutes").sum() * 36).alias("reb36"),
            (pl.col("game_date").max() >= cutoff).alias("active"),
        )
    )
    bigs = (
        roster.select("player_id", "team_now")
        .join(pos, on="player_id")
        .filter(pl.col("position").is_in(BIG_POSITIONS))
        .join(recent, on="player_id")
        .with_columns(
            (pl.col("position").is_in(CENTRE_POSITIONS) | (pl.col("reb36") >= CENTRE_REB36))
            .alias("is_centre"),
            pl.col("player_name").map_elements(player_key, return_dtype=pl.String).alias("key"),
        )
        .sort(["active", "recent_min"], descending=[True, True])
    )
    starters = (
        bigs.filter(pl.col("is_centre"))
        .group_by("team_now", maintain_order=True)
        .first()
        .select("team_now", pl.col("player_id").alias("starter_id"))
    )
    return bigs.join(starters, on="team_now", how="left").with_columns(
        (pl.col("player_id") == pl.col("starter_id")).alias("is_starter")
    )


def is_new_absence(pg: pl.DataFrame, starter_id: int, team: str, game_day: date) -> bool:
    """True if the starter played his team's most recent game before game_day.

    Fresh absences are where the edge is; long-term ones are already in the price.
    """
    before = pg.filter(pl.col("game_date") < game_day)
    team_games = before.filter(pl.col("team") == team)
    if team_games.is_empty():
        return True
    last_team_game = team_games["game_date"].max()
    played = before.filter(pl.col("player_id") == starter_id)["game_date"]
    return played.len() > 0 and played.max() == last_team_game


def mixed_prices(name: str, p_out: float, combos: list[str], reb_mult: float = 1.0,
                 ast_mult: float = 1.0) -> pl.DataFrame | None:
    """Fair prices for a backup, mixing starter-plays and starter-sits by p_out.

    reb_mult scales his rebound rate for tonight's opponent (rebounding.py).
    """
    hist, _ = player_history(name)
    fit = fit_player(hist)
    if fit is None:
        return None
    fit.reb_rate = fit.reb_rate * reb_mult
    fit.ast_rate = fit.ast_rate * ast_mult
    parsed = [parse_combo(c) for c in combos]
    sits = combo_probs(simulate_stats(fit, True, 30_000, seed=1), parsed)
    plays = combo_probs(simulate_stats(fit, False, 30_000, seed=2), parsed)
    p = p_out * sits["p"] + (1 - p_out) * plays["p"]
    p_indep = p_out * sits["p_indep"] + (1 - p_out) * plays["p_indep"]
    return pl.DataFrame({
        "combo": sits["combo"], "p": p, "fair": 1 / p, "fair_if_sits": sits["fair"],
        "fair_if_plays": plays["fair"], "fair_indep": 1 / p_indep,
    })


def _luck(luck: pl.DataFrame | None, player_id: int) -> tuple[float, float]:
    """(AST, REB) luck multipliers for a player; 1.0 when unknown."""
    if luck is None:
        return 1.0, 1.0
    row = luck.filter(pl.col("player_id") == player_id)
    return (row["luck_ast"][0], row["luck_reb"][0]) if row.height else (1.0, 1.0)


def _opponent(matchup: str, team: str) -> str:
    """'MEM@POR', 'POR' -> 'MEM'."""
    away, home = matchup.split("@")
    return away if team == home else home


def minutes_unlocked(report: pl.DataFrame, pg: pl.DataFrame, roster: pl.DataFrame,
                     pos: pl.DataFrame, skip_ids: set[int],
                     reb_table: pl.DataFrame | None = None,
                     luck: pl.DataFrame | None = None) -> list[str]:
    """Players projected to gain UNLOCK_MIN_SHIFT+ minutes from today's Out/Doubtful teammates."""
    eff_raw = pair_effects(pg, pos)
    pri = priors(eff_raw)
    eff = shrunk_effects(eff_raw, pri)
    recent = (pg.sort("game_date").group_by("player_id").tail(RECENT_GAMES)
              .group_by("player_id").agg(pl.col("minutes").mean().alias("recent_min")))
    people = roster.join(recent, on="player_id").with_columns(
        pl.col("player_name").map_elements(player_key, return_dtype=pl.String).alias("player_key")
    )
    listed = report.select("team", "player_key", "status")
    # Out/Doubtful today, matched to player ids on the current roster.
    out_today = (
        listed.filter(pl.col("status").is_in(ALERT_STATUSES))
        .join(people, left_on=["team", "player_key"], right_on=["team_now", "player_key"])
        .filter(pl.col("recent_min") >= MIN_MATE_MINUTES)
        .select(pl.col("team").alias("team_id"), pl.col("player_id").alias("mate_id"),
                pl.col("recent_min").alias("mate_min"), pl.lit("today").alias("game_id"))
    )
    teams_today = report["team"].drop_nulls().unique().implode()
    candidates = (
        people.filter(pl.col("team_now").is_in(teams_today), ~pl.col("player_id").is_in(list(skip_ids)))
        .join(listed.filter(pl.col("status").is_in(ALERT_STATUSES)),
              left_on=["team_now", "player_key"], right_on=["team", "player_key"], how="anti")
        .select("player_id", "player_name", pl.col("team_now").alias("team_id"), "recent_min",
                pl.lit("today").alias("game_id"))
    )
    today = context_effects(candidates, out_today, eff, pri, pos).join(
        candidates.drop("game_id"), on="player_id"
    ).filter(pl.col("shift_min") >= UNLOCK_MIN_SHIFT).sort("shift_min", descending=True)
    if today.is_empty():
        return ["None."]
    absent_hist = game_absences(pg)
    combos = [parse_combo(c) for c in UNLOCK_COMBOS]
    out = []
    for r in today.iter_rows(named=True):
        hist = pg.filter(pl.col("player_id") == r["player_id"]).sort("game_date").tail(HISTORY_GAMES)
        hist = hist.join(context_effects(hist, absent_hist, eff, pri, pos), on=["player_id", "game_id"])
        fit = fit_context(hist)
        if fit is None:
            out.append(f"- {r['player_name']} ({r['team_id']}): +{r['shift_min']:.1f} min, too little history to price")
            continue
        mult = {s: r[f"mult_{s}"] for s in RATES}
        matchup = report.filter(pl.col("team") == r["team_id"])["matchup"][0]
        opp = _opponent(matchup, r["team_id"])
        rm = live_rebound_multiplier(pg, pos, r["player_id"], opp, reb_table)
        la, lr = _luck(luck, r["player_id"])
        mult["reb"] *= rm * lr
        mult["ast"] *= la
        sim_today = simulate_context(fit, r["shift_min"], mult, 30_000, seed=1)
        sim_full = simulate_context(fit, 0.0, None, 30_000, seed=2)
        p_today, p_full = combo_probs(sim_today, combos), combo_probs(sim_full, combos)
        who = out_today.filter(pl.col("team_id") == r["team_id"]).join(
            roster.select(pl.col("player_id").alias("mate_id"), "player_name"), on="mate_id")
        rates = ", ".join(f"{s.upper()[0]}PM {(r[f'mult_{s}'] - 1) * 100:+.0f}%" for s in RATES)
        rates += f"; vs {opp} rebounds x{rm:.2f}; luck AST x{la:.2f} REB x{lr:.2f}"
        out += [
            (f"- **{r['player_name']}** ({r['team_id']}): full-strength "
             f"{sim_full['minutes'].mean():.0f} → projected {sim_today['minutes'].mean():.0f} min "
             f"(+{r['shift_min']:.1f}); {rates}. Out: {', '.join(who['player_name'].to_list())}"),
            "",
            "| Combo | Fair today | Fair at full strength | Independent legs today |",
            "|---|---|---|---|",
            *[f"| {a['combo']} | ${a['fair']:.2f} | ${b['fair']:.2f} | ${a['fair_indep']:.2f} |"
              for a, b in zip(p_today.iter_rows(named=True), p_full.iter_rows(named=True), strict=True)],
            "",
        ]
    return out


def build_sheet(game_day: date) -> str:
    report = report_for_date(game_day)
    if report.is_empty():
        return f"# Sheet {game_day}\n\nNo injury report found for {game_day} (ET).\n"
    report = report.filter(pl.col("game_date") == game_day).with_columns(
        pl.col("team_name").replace_strict(TEAM_ABBR, default=None).alias("team")
    )
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    roster = parse_player_index(get_player_index("2026-27"))
    pos_all = positions()
    dc = depth_chart(pg, roster, pos_all)
    reb_table = live_multiplier_table(pg, pos_all)
    try:
        luck = live_luck()
    except (OSError, ValueError, KeyError) as err:  # tracking endpoint down: price without luck
        print(f"luck adjustment unavailable: {err!r}")
        luck = pl.DataFrame(schema={"player_id": pl.Int64, "luck_ast": pl.Float64, "luck_reb": pl.Float64})
    teams_today = report["team"].drop_nulls().unique().to_list()
    status = report.select("team", "player_key", "status", "reason")

    lines = [
        f"# Game-day sheet: {game_day}",
        (f"Injury report: {report['report_time'].max()} ET. "
         f"P(sits) by status (calibrated 2025-26): "
         f"{', '.join(f'{k} {v:.0%}' for k, v in P_OUT.items())}."),
        "",
    ]
    not_filed = report.filter(pl.col("status") == "Not submitted")["team"].drop_nulls().to_list()
    if not_filed:
        lines += [f"**Not yet submitted:** {', '.join(sorted(not_filed))} (re-run later)", ""]

    sections = {"alert": [], "watch": []}
    priced_ids: set[int] = set()
    for team in sorted(teams_today):
        chart = dc.filter(pl.col("team_now") == team)
        starter = chart.filter(pl.col("is_starter"))
        if starter.is_empty():
            continue
        s = starter.row(0, named=True)
        st = status.filter((pl.col("team") == team) & (pl.col("player_key") == s["key"]))
        s_status = st["status"][0] if st.height else "Not listed"
        kind = "alert" if s_status in ALERT_STATUSES else "watch" if s_status in WATCH_STATUSES else None
        if kind is None:
            continue
        p_out = P_OUT[s_status]
        matchup = report.filter(pl.col("team") == team)["matchup"][0]
        new = is_new_absence(pg, s["player_id"], team, game_day)
        reason = f" ({st['reason'][0]})" if st.height and st["reason"][0] else ""
        freshness = "**NEW: played last game**" if new else "ongoing absence (likely priced in)"
        block = [f"### {team} ({matchup}): {s['player_name']} **{s_status}**{reason} — {freshness}"]
        priced = 0
        for b in chart.filter(~pl.col("is_starter")).iter_rows(named=True):
            if priced >= BACKUPS_PER_TEAM:
                break
            bst = status.filter((pl.col("team") == team) & (pl.col("player_key") == b["key"]))
            b_status = bst["status"][0] if bst.height else "Not listed"
            if b_status in ALERT_STATUSES:
                block.append(f"- {b['player_name']}: {b_status} himself, skipped")
                continue
            priced += 1
            priced_ids.add(b["player_id"])
            opp = _opponent(matchup, team)
            rm = live_rebound_multiplier(pg, pos_all, b["player_id"], opp, reb_table)
            la, lr = _luck(luck, b["player_id"])
            prices = mixed_prices(b["player_name"], p_out, COMBOS, rm * lr, la)
            if prices is None:
                block.append(f"- {b['player_name']}: not enough history to price")
                continue
            block += [
                (f"- **{b['player_name']}** ({b['position']}, {b['recent_min']:.0f} min recently"
                 f"{', ' + b_status if b_status != 'Not listed' else ''}; "
                 f"vs {opp} rebounds x{rm:.2f}; luck AST x{la:.2f} REB x{lr:.2f})"),
                "",
                "| Combo | Fair | If starter sits | If plays | Independent legs |",
                "|---|---|---|---|---|",
                *[f"| {r['combo']} | ${r['fair']:.2f} | ${r['fair_if_sits']:.2f} | "
                  f"${r['fair_if_plays']:.2f} | ${r['fair_indep']:.2f} |"
                  for r in prices.iter_rows(named=True)],
                "",
            ]
        sections[kind].append((not new, "\n".join(block)))

    lines += ["## ALERTS: starting centre Out / Doubtful", ""]
    # New absences first: that's where the book is most likely to be slow.
    lines += [b for _, b in sorted(sections["alert"], key=lambda x: x[0])] or ["None."]
    lines += ["", "## Watch: starting centre Questionable", ""]
    lines += [b for _, b in sorted(sections["watch"], key=lambda x: x[0])] or ["None."]
    lines += ["", f"## Minutes unlocked: everyone else (+{UNLOCK_MIN_SHIFT:.0f} min or more)", ""]
    alert_backups = {b for b in priced_ids}
    lines += minutes_unlocked(report, pg, roster, pos_all, alert_backups, reb_table, luck)
    lines += [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, default=datetime.now(ET).date())
    args = ap.parse_args()
    sheet = build_sheet(args.date)
    out_dir = config.DATA_PROCESSED / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sheet_{args.date.isoformat()}.md"
    path.write_text(sheet)
    print(sheet)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
