"""What each defence allows by shot type (stats.nba.com LeagueDashOppPtShot), cached and throttled.

general_range "Catch and Shoot" = opponents' catch-and-shoot attempts and makes against this team; "Overall"
for the totals. Raw responses go to data/raw/nba_stats/oppptshot/<season>_<range>_<date>.json (one per day,
never overwritten).

    uv run python -m nba_ladders.sources.opp_shots [--season 2026-27] [--team MIA]
"""

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from nba_ladders.sources.nba_stats import RAW_ROOT, _throttle

RAW_DIR = RAW_ROOT / "oppptshot"
RANGES = {"Catch and Shoot": "cs", "Overall": "all"}


def _fetch(season: str, general_range: str) -> dict:
    from nba_api.stats.endpoints import leaguedashoppptshot

    _throttle()
    return leaguedashoppptshot.LeagueDashOppPtShot(season=season, general_range_nullable=general_range,
                                                   per_mode_simple="Totals", timeout=30).get_dict()


def get(season: str, general_range: str, as_of: date | None = None, raw_dir: Path = RAW_DIR, fetch=_fetch) -> dict:
    as_of = as_of or date.today()
    path = raw_dir / f"{season}_{RANGES[general_range]}_{as_of.isoformat()}.json"
    if path.exists():
        return json.loads(path.read_text())
    done = sorted(raw_dir.glob(f"{season}_{RANGES[general_range]}_*.json")) if season < "2026-27" else []
    if done:  # a completed season: any snapshot will do
        return json.loads(done[-1].read_text())
    payload = fetch(season, general_range)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def parse(payload: dict) -> pl.DataFrame:
    rs = payload["resultSets"][0]
    df = pl.DataFrame(rs["rowSet"], schema=rs["headers"], orient="row")
    return df.select(pl.col("TEAM_ABBREVIATION").alias("team"), pl.col("GP").alias("gp"),
                     pl.col("FG3M").alias("fg3m"), pl.col("FG3A").alias("fg3a"))


def table(season: str, as_of: date | None = None) -> pl.DataFrame:
    """Per defence: catch-and-shoot 3PA / 3PM allowed per game, C&S share of all 3PA allowed, and C&S 3P% allowed,
    each relative to the league (1.00 = average)."""
    cs, al = parse(get(season, "Catch and Shoot", as_of)), parse(get(season, "Overall", as_of))
    d = cs.join(al.select("team", pl.col("fg3a").alias("fg3a_all")), on="team").with_columns(
        (pl.col("fg3a") / pl.col("gp")).alias("cs3a_pg"), (pl.col("fg3m") / pl.col("gp")).alias("cs3m_pg"),
        (pl.col("fg3m") / pl.col("fg3a")).alias("cs3p"), (pl.col("fg3a") / pl.col("fg3a_all")).alias("cs_share"))
    lg = {c: d[c].mean() for c in ("cs3a_pg", "cs3m_pg", "cs3p", "cs_share")}
    return d.with_columns(*[(pl.col(c) / lg[c]).alias(f"{c}_rel") for c in lg]).sort("cs3m_pg", descending=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--team")
    a = ap.parse_args()
    t = table(a.season)
    with pl.Config(tbl_rows=32, tbl_hide_column_data_types=True, float_precision=2):
        print(t.select("team", "gp", "cs3a_pg", "cs3m_pg", "cs3p", "cs_share", "cs3a_pg_rel", "cs3m_pg_rel", "cs3p_rel"))
        if a.team:
            r = t.with_row_index("rank", 1).filter(pl.col("team") == a.team)
            print(f"\n{a.team}: rank {r['rank'][0]}/30 in catch-and-shoot 3PM allowed per game")


if __name__ == "__main__":
    main()
