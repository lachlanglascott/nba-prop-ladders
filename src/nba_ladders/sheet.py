"""Daily game-day sheet: injury report -> starting centres out -> backup bigs' fair ladders.

    uv run python -m nba_ladders.sheet                 # today's games (US Eastern date)
    uv run python -m nba_ladders.sheet --date 2026-10-24

Alerts (starting centre Out/Doubtful) come first; Questionable centres are priced as a mix
of "plays" and "sits". Writes data/processed/sheets/sheet_<date>.md.
"""

import argparse
from datetime import date, datetime

import polars as pl
from nba_api.stats.static import teams

from nba_ladders import config
from nba_ladders.price import player_history
from nba_ladders.scenarios import CENTRE_REB36, positions
from nba_ladders.screens import BIG_POSITIONS, CENTRE_POSITIONS, parse_player_index
from nba_ladders.sim import combo_probs, fit_player, parse_combo, simulate_stats
from nba_ladders.sources.injury_report import ET, player_key, report_for_date
from nba_ladders.sources.nba_stats import get_player_index

# Assumed P(doesn't play) by listed status. Placeholders until calibrated against history.
P_OUT = {"Out": 1.0, "Doubtful": 0.8, "Questionable": 0.4, "Probable": 0.05, "Available": 0.0}
ALERT_STATUSES = {"Out", "Doubtful"}
WATCH_STATUSES = {"Questionable"}
RECENT_GAMES = 15
BACKUPS_PER_TEAM = 2
COMBOS = ["8p/8r", "10p/10r", "12p/12r", "15p/12r", "10p/8r/2a", "12p/10r/3a"]
TEAM_ABBR = {t["full_name"].replace(" ", ""): t["abbreviation"] for t in teams.get_teams()}


def depth_chart(pg: pl.DataFrame, roster: pl.DataFrame, pos: pl.DataFrame) -> pl.DataFrame:
    """Bigs per current team, ranked by recent minutes, with the starting centre flagged.

    roster: player_id -> team_now (current player index). Recent minutes are each player's
    last RECENT_GAMES games on any team, so offseason moves are handled.
    """
    recent = (
        pg.sort("game_date")
        .group_by("player_id")
        .tail(RECENT_GAMES)
        .group_by("player_id")
        .agg(
            pl.col("player_name").last(),
            pl.col("minutes").mean().alias("recent_min"),
            (pl.col("reb").sum() / pl.col("minutes").sum() * 36).alias("reb36"),
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
        .sort("recent_min", descending=True)
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


def mixed_prices(name: str, p_out: float, combos: list[str]) -> pl.DataFrame | None:
    """Fair prices for a backup, mixing starter-plays and starter-sits by p_out."""
    hist, _ = player_history(name)
    fit = fit_player(hist)
    if fit is None:
        return None
    parsed = [parse_combo(c) for c in combos]
    sits = combo_probs(simulate_stats(fit, True, 30_000, seed=1), parsed)
    plays = combo_probs(simulate_stats(fit, False, 30_000, seed=2), parsed)
    p = p_out * sits["p"] + (1 - p_out) * plays["p"]
    p_indep = p_out * sits["p_indep"] + (1 - p_out) * plays["p_indep"]
    return pl.DataFrame({
        "combo": sits["combo"], "p": p, "fair": 1 / p, "fair_if_sits": sits["fair"],
        "fair_if_plays": plays["fair"], "fair_indep": 1 / p_indep,
    })


def build_sheet(game_day: date) -> str:
    report = report_for_date(game_day)
    if report.is_empty():
        return f"# Sheet {game_day}\n\nNo injury report found for {game_day} (ET).\n"
    report = report.filter(pl.col("game_date") == game_day).with_columns(
        pl.col("team_name").replace_strict(TEAM_ABBR, default=None).alias("team")
    )
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    roster = parse_player_index(get_player_index("2026-27"))
    dc = depth_chart(pg, roster, positions())
    teams_today = report["team"].drop_nulls().unique().to_list()
    status = report.select("team", "player_key", "status", "reason")

    lines = [
        f"# Game-day sheet: {game_day}",
        f"Injury report: {report['report_time'].max()} ET. "
        f"Assumed P(sits): {', '.join(f'{k} {v:.0%}' for k, v in P_OUT.items())}.",
        "",
    ]
    not_filed = report.filter(pl.col("status") == "Not submitted")["team"].drop_nulls().to_list()
    if not_filed:
        lines += [f"**Not yet submitted:** {', '.join(sorted(not_filed))} (re-run later)", ""]

    sections = {"alert": [], "watch": []}
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
        block = [f"### {team} ({matchup}): {s['player_name']} **{s_status}**"
                 f"{' (' + st['reason'][0] + ')' if st.height and st['reason'][0] else ''}"
                 f" — {'**NEW: played last game**' if new else 'ongoing absence (likely priced in)'}"]
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
            prices = mixed_prices(b["player_name"], p_out, COMBOS)
            if prices is None:
                block.append(f"- {b['player_name']}: not enough history to price")
                continue
            block += [
                f"- **{b['player_name']}** ({b['position']}, {b['recent_min']:.0f} min recently"
                f"{', ' + b_status if b_status != 'Not listed' else ''})",
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
    lines += ["", "_Other injuries (non-centre) are not modelled yet._", ""]
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
