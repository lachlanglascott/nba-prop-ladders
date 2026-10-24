"""Cached, throttled access to stats.nba.com via nba_api.

Raw responses are written once to data/raw/nba_stats/ and never overwritten. A completed
season is fetched once; the current season gets a new dated snapshot per fetch day.
"""

import json
import time
from datetime import date
from pathlib import Path

import truststore

from nba_ladders import config

# Use the OS certificate store (needed behind TLS-inspecting proxies; harmless elsewhere).
truststore.inject_into_ssl()

RAW_DIR = config.DATA_RAW / "nba_stats" / "leaguegamelog"
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    wait = config.NBA_API_MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _fetch_league_gamelog(season: str, season_type: str) -> dict:
    from nba_api.stats.endpoints import leaguegamelog

    _throttle()
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
        timeout=60,
    )
    return resp.get_dict()


def snapshot_path(season: str, season_type: str, as_of: date, raw_dir: Path = RAW_DIR) -> Path:
    slug = season_type.lower().replace(" ", "_")
    return raw_dir / f"{season}_{slug}_{as_of.isoformat()}.json"


def latest_snapshot(season: str, season_type: str, raw_dir: Path = RAW_DIR) -> Path | None:
    slug = season_type.lower().replace(" ", "_")
    files = sorted(raw_dir.glob(f"{season}_{slug}_*.json"))
    return files[-1] if files else None


def get_league_gamelog(
    season: str,
    season_type: str = "Regular Season",
    *,
    refresh: bool = False,
    raw_dir: Path = RAW_DIR,
    fetch=_fetch_league_gamelog,
    today: date | None = None,
) -> dict:
    """Return the raw player game log for a season, using the cache when possible.

    refresh=True fetches a new dated snapshot (use for the in-progress season); an existing
    snapshot for the same day is reused rather than overwritten.
    """
    today = today or date.today()
    existing = latest_snapshot(season, season_type, raw_dir)
    if existing and not refresh:
        return json.loads(existing.read_text())

    path = snapshot_path(season, season_type, today, raw_dir)
    if path.exists():
        return json.loads(path.read_text())

    payload = fetch(season, season_type)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload
