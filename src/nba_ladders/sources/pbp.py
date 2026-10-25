"""Cached, throttled play-by-play (PlayByPlayV3) from stats.nba.com.

One raw JSON per game at data/raw/nba_stats/playbyplayv3/<game_id>.json. Completed games are
fetched once and never re-fetched or overwritten. Run as a module to pull a season:

    uv run python -m nba_ladders.sources.pbp --season-id 22025
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path

import polars as pl
import requests

from nba_ladders import config
from nba_ladders.sources import nba_stats  # noqa: F401  (truststore injection + throttle)
from nba_ladders.sources.nba_stats import RAW_ROOT, _throttle

PBP_DIR = RAW_ROOT / "playbyplayv3"
BACKOFF_SECONDS = (5, 20, 60)

FOUL_SCHEMA = {
    "game_id": pl.String,
    "action_number": pl.Int64,
    "period": pl.Int64,
    "clock": pl.String,
    "seconds_elapsed": pl.Float64,
    "player_id": pl.Int64,
    "player_name": pl.String,
    "team_id": pl.Int64,
    "team": pl.String,
    "sub_type": pl.String,
    "player_pf_count": pl.Int64,
    "description": pl.String,
}

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
# "(P2.T4)", "(P1)" on offensive fouls, "(P5.PN)" in the penalty; technicals show "(P0.T0)".
_PF_COUNT_RE = re.compile(r"\(P(\d+)[.)]")


def _fetch_pbp(game_id: str) -> dict:
    from nba_api.stats.endpoints import playbyplayv3

    _throttle()
    return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30).get_dict()


def pbp_path(game_id: str, raw_dir: Path = PBP_DIR) -> Path:
    return raw_dir / f"{game_id}.json"


def _is_complete(payload: dict) -> bool:
    actions = payload.get("game", {}).get("actions") or []
    return any(a.get("actionType") == "game" or "End of 4th" in a.get("description", "")
               for a in actions) or len(actions) > 300


def get_pbp(
    game_id: str,
    *,
    raw_dir: Path = PBP_DIR,
    fetch: Callable[[str], dict] = _fetch_pbp,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Return raw PBP for a game from cache, else fetch (with retry/backoff) and cache it."""
    path = pbp_path(game_id, raw_dir)
    if path.exists():
        return json.loads(path.read_text())
    last_err: Exception | None = None
    for attempt, wait in enumerate((0, *BACKOFF_SECONDS)):
        if wait:
            sleep(wait)
        try:
            payload = fetch(game_id)
        # timeouts, 429s / block pages (non-JSON -> ValueError), connection resets
        except (requests.RequestException, OSError, ValueError, KeyError) as err:
            last_err = err
            print(f"  {game_id} attempt {attempt + 1} failed: {err!r}", file=sys.stderr)
            continue
        if not payload.get("game", {}).get("actions"):
            last_err = ValueError(f"empty actions for {game_id}")
            continue
        if _is_complete(payload):  # only cache finished games
            raw_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.part")
            tmp.write_text(json.dumps(payload))
            tmp.rename(path)
        return payload
    raise RuntimeError(f"failed to fetch PBP for {game_id}") from last_err


def clock_to_seconds_left(clock: str) -> float:
    m = _CLOCK_RE.match(clock or "")
    return int(m.group(1)) * 60 + float(m.group(2)) if m else float("nan")


def seconds_elapsed(period: int, clock: str) -> float:
    """Game seconds elapsed at the event (12-min quarters, 5-min OTs)."""
    left = clock_to_seconds_left(clock)
    if period <= 4:
        return (period - 1) * 720 + (720 - left)
    return 4 * 720 + (period - 5) * 300 + (300 - left)


def is_personal_foul(action: dict) -> bool:
    """Personal fouls (incl. flagrant) carry a '(Pn.' player count; technicals do not."""
    if action.get("actionType") != "Foul" or not action.get("personId"):
        return False
    sub = (action.get("subType") or "").lower()
    if "technical" in sub or "defensive 3" in sub or "delay" in sub:
        return False
    m = _PF_COUNT_RE.search(action.get("description") or "")
    return bool(m) and int(m.group(1)) > 0


def parse_fouls(payload: dict) -> pl.DataFrame:
    """Tidy personal-foul table for one game."""
    game = payload["game"]
    rows = []
    for a in game["actions"]:
        if not is_personal_foul(a):
            continue
        cnt = _PF_COUNT_RE.search(a["description"])
        rows.append({
            "game_id": game["gameId"],
            "action_number": a["actionNumber"],
            "period": a["period"],
            "clock": a["clock"],
            "seconds_elapsed": seconds_elapsed(a["period"], a["clock"]),
            "player_id": a["personId"],
            "player_name": a["playerName"],
            "team_id": a["teamId"],
            "team": a["teamTricode"],
            "sub_type": a["subType"],
            "player_pf_count": int(cnt.group(1)) if cnt else None,
            "description": a["description"],
        })
    return pl.DataFrame(rows, schema=FOUL_SCHEMA)


STARTER_SCHEMA = {"game_id": pl.String, "team": pl.String, "player_id": pl.Int64}
_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+")


def _norm_name(name: str | None) -> str:
    """'Valančiūnas' and 'Valanciunas' must match (sub descriptions drop diacritics)."""
    folded = unicodedata.normalize("NFKD", name or "")
    return "".join(c for c in folded if not unicodedata.combining(c)).lower().strip()


def parse_starters(payload: dict) -> pl.DataFrame:
    """Starters per team, inferred from Q1: a player is a starter if his first Q1 action
    comes before anyone of that name was subbed in for his team.

    Substitution rows carry the player going OUT as personId; the incoming player appears
    only by (short) name in the description, so we match on playerName within the team.
    """
    game = payload["game"]
    subbed_in: set[tuple[str, str]] = set()  # exact (team, name) from "SUB: <name> FOR"
    subbed_last: set[tuple[str, str]] = set()  # "K. Williams" -> "williams" (disambiguated)
    seen: set[int] = set()
    cands: list[tuple[str, int, str]] = []
    for a in game["actions"]:
        if a.get("period") != 1:
            continue
        team, pid = a.get("teamTricode") or "", a.get("personId") or 0
        name = _norm_name(a.get("playerName"))
        name_i = _norm_name(a.get("playerNameI"))  # "k. williams" when surnames clash
        if a.get("actionType") == "Substitution":
            m = _SUB_RE.match(a.get("description") or "")
            if m:
                inc = _norm_name(m.group(1))
                subbed_in.add((team, inc))
                subbed_last.add((team, inc.split(". ", 1)[-1]))
        if not pid or not team or pid in seen:
            continue
        seen.add(pid)
        if (team, name) not in subbed_in and (team, name_i) not in subbed_in:
            cands.append((team, pid, name))
    rows = []
    for team in {t for t, _, _ in cands}:
        mine = [c for c in cands if c[0] == team]
        if len(mine) > 5:  # shared surname: drop anyone matching an abbreviated sub-in,
            kept = [c for c in mine if (team, c[2]) not in subbed_last]
            mine = kept if len(kept) == 5 else mine  # ...but only if that resolves it
        rows += [{"game_id": game["gameId"], "team": t, "player_id": p} for t, p, _ in mine]
    return pl.DataFrame(rows, schema=STARTER_SCHEMA)


def pull_season(
    game_ids: Iterable[str], *, raw_dir: Path = PBP_DIR, max_consecutive_failures: int = 3
) -> tuple[int, int, list[str]]:
    """Fetch every uncached game. Stops early if the source looks like it is blocking us.

    Returns (n_cached_before, n_fetched, failed_ids).
    """
    ids = sorted(set(game_ids))
    todo = [g for g in ids if not pbp_path(g, raw_dir).exists()]
    before = len(ids) - len(todo)
    print(f"{len(ids)} games, {before} cached, {len(todo)} to fetch", flush=True)
    fetched, failed, streak = 0, [], 0
    for i, gid in enumerate(todo, 1):
        try:
            get_pbp(gid, raw_dir=raw_dir)
            fetched += 1
            streak = 0
        except RuntimeError as err:
            failed.append(gid)
            streak += 1
            print(f"FAILED {gid}: {err.__cause__!r}", flush=True)
            if streak >= max_consecutive_failures:
                print(f"STOPPING: {streak} consecutive failures (possible block)", flush=True)
                break
        if i % 50 == 0:
            print(f"progress {i}/{len(todo)} fetched={fetched} failed={len(failed)}", flush=True)
    print(f"done: cached_before={before} fetched={fetched} failed={len(failed)}", flush=True)
    return before, fetched, failed


def load_fouls(game_ids: Iterable[str], raw_dir: Path = PBP_DIR) -> pl.DataFrame:
    """Parse fouls from cached PBP only (no network)."""
    frames = [
        parse_fouls(json.loads(p.read_text()))
        for g in sorted(set(game_ids))
        if (p := pbp_path(g, raw_dir)).exists()
    ]
    return pl.concat(frames) if frames else pl.DataFrame(schema=FOUL_SCHEMA)


def load_starters(game_ids: Iterable[str], raw_dir: Path = PBP_DIR) -> pl.DataFrame:
    frames = [
        parse_starters(json.loads(p.read_text()))
        for g in sorted(set(game_ids))
        if (p := pbp_path(g, raw_dir)).exists()
    ]
    return pl.concat(frames) if frames else pl.DataFrame(schema=STARTER_SCHEMA)


def _season_game_ids(season_id: str) -> list[str]:
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    return pg.filter(pl.col("season_id") == season_id)["game_id"].unique().to_list()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-id", default="22025")
    ap.add_argument("--build", action="store_true", help="write processed fouls parquet only")
    args = ap.parse_args()
    ids = _season_game_ids(args.season_id)
    if not args.build:
        pull_season(ids)
    fouls = load_fouls(ids)
    out = config.DATA_PROCESSED / f"fouls_{args.season_id}.parquet"
    fouls.write_parquet(out)
    print(f"wrote {out} ({fouls.height} fouls, {fouls['game_id'].n_unique()} games)")
    starters = load_starters(ids)
    out = config.DATA_PROCESSED / f"starters_{args.season_id}.parquet"
    starters.write_parquet(out)
    print(f"wrote {out} ({starters.height} starter rows)")


if __name__ == "__main__":
    main()
