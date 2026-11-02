"""Who actually plays: carry-forward of last game's absences, and uncertain teammates.

Measured on 2025-26 and 2026-27 injury reports (~5 PM ET, 13.7k listings) vs box scores, and
on 2023-26 game logs (20+ mpg players):
- Out / Doubtful sit 99.9% / 99%; Questionable 53% (2025-26) and 60% (2026-27, n=48) -> 0.55.
- Out at the previous game (not surgery): sits next game 85-89% for an injury, 51-61% when
  the reason was "management" or rest.
- No report yet: missed 1 / 2 / 3 / 4+ straight games -> misses the next 42% / 65-69% / 71-75% / 83-86%.
- Team filed and he's NOT listed, but he missed the last game: still sits 26-35%.
- Unlisted and played last game: 2-3% DNP (late scratches, coach's decisions).
"""

from datetime import date

import polars as pl

from nba_ladders.sources.injury_report import player_key

P_AFTER_OUT = {"injury": 0.87, "management": 0.56, "surgery": 0.99}
P_STREAK = {1: 0.42, 2: 0.67, 3: 0.73, 4: 0.84}  # no report: by consecutive games missed
P_UNLISTED_MISSED = 0.30  # team filed, not listed, but missed the last game
OUT_AT = 0.70  # treat as Out at or above this; otherwise Questionable with its own P(sits)
MIN_MPG = 20.0


def _reason_class(reason: str) -> str:
    r = (reason or "").lower()
    if any(w in r for w in ("surgery", "repair", "fracture", "tear", "acl", "notwithteam")):
        return "surgery"
    return "management" if ("management" in r or "rest" in r) else "injury"


def carry_forward(report: pl.DataFrame, pg: pl.DataFrame, day: date, prev_report=None) -> pl.DataFrame:
    """Rows to add to tonight's report for rotation players who missed their team's last game
    and aren't on tonight's report. prev_report(d) returns the report for an earlier date."""
    cur = pg.filter(pl.col("season_id") == pg.filter(pl.col("game_date") < day)["season_id"].max(),
                    pl.col("game_date") < day)
    teams = report["team"].drop_nulls().unique().to_list()
    unfiled = set(report.filter(pl.col("status") == "Not submitted")["team"].drop_nulls())
    listed = set(report["player_key"].drop_nulls())
    rot = cur.group_by("player_name", "team").agg(pl.col("minutes").mean().alias("mpg"), pl.len().alias("gp"),
                                                  pl.col("game_date").max().alias("my_last"))
    rot = rot.filter(pl.col("team").is_in(teams), pl.col("mpg") >= MIN_MPG, pl.col("gp") >= 2)
    games = cur.select("team", "game_date").unique().sort("game_date")
    rows, prev_cache = [], {}
    for r in rot.iter_rows(named=True):
        tdates = games.filter(pl.col("team") == r["team"])["game_date"].to_list()
        if not tdates or r["my_last"] == tdates[-1]:
            continue  # played the last game
        key = player_key(r["player_name"])
        if key in listed:
            continue
        streak = sum(1 for d in tdates if d > r["my_last"])
        if r["team"] in unfiled:
            p, why = P_STREAK[min(streak, 4)], f"missed {streak} straight"
            last = tdates[-1]
            if prev_report is not None:
                if last not in prev_cache:
                    prev_cache[last] = prev_report(last)
                pr = prev_cache[last]
                hit = pr.filter(pl.col("player_key") == key, pl.col("status").is_in(["Out", "Doubtful"])) if pr is not None else None
                if hit is not None and hit.height:
                    cls = _reason_class(hit["reason"][0])
                    p, why = P_AFTER_OUT[cls], f"Out last game ({cls}: {hit['reason'][0]})"
            why += "; no report yet"
        else:
            p, why = P_UNLISTED_MISSED, f"missed last game but not on tonight's report (missed {streak})"
        rows.append({"team": r["team"], "player": r["player_name"], "player_key": key,
                     "status": "Out" if p >= OUT_AT else "Questionable", "p_sit": p,
                     "reason": f"Carry-forward: {why}, P(sits) {p:.0%}"})
    return pl.DataFrame(rows, schema={"team": pl.String, "player": pl.String, "player_key": pl.String,
                                      "status": pl.String, "p_sit": pl.Float64, "reason": pl.String})
