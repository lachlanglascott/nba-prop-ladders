"""The Odds API (v4): NBA player prop prices, with a hard credit budget.

    uv run python -m nba_ladders.sources.odds_api coverage                 # ~1 credit
    uv run python -m nba_ladders.sources.odds_api snapshot --max-credits 40
    uv run python -m nba_ladders.sources.odds_api historical --date 2026-03-15T23:00:00Z --max-credits 100

Key: ODDS_API_KEY in the environment or in .env at the repo root (git-ignored). It is never
printed or logged. Costs (docs, Oct 2026): event odds = markets returned x regions; historical
event odds = 10 x markets returned x regions; /events is free; empty responses are free.
Every call is estimated before it's made and refused if it would exceed the run's budget or
leave fewer than RESERVE credits in the account. Raw responses are cached under
data/raw/odds_api/ and never overwritten.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests
import truststore

from nba_ladders import config

truststore.inject_into_ssl()

BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
RAW_DIR = config.DATA_RAW / "odds_api"
RESERVE = 50  # never let a run take the account below this many credits
AU_BASE = "sportsbet"
US_REFERENCE = ["fanduel", "draftkings"]  # fanatics dropped for picklebet_au (keeps 10 books = 1 cost unit)
AU_REFERENCE = ["tab", "dabble_au", "betr_au", "neds", "ladbrokes_au", "pointsbetau", "picklebet_au"]  # legs only (no SGM in API)
DEFAULT_BOOKMAKERS = [AU_BASE, *AU_REFERENCE, *US_REFERENCE]  # <= 10 books = 1 region of cost
DEFAULT_MARKETS = [
    "player_points_alternate", "player_rebounds_alternate", "player_assists_alternate",
    "player_points_rebounds_alternate", "player_points_assists_alternate",
    "player_rebounds_assists_alternate", "player_threes_alternate",
]
STAT_OF_MARKET = {
    "player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
    "player_points_rebounds": "pts+reb", "player_points_assists": "pts+ast",
    "player_rebounds_assists": "reb+ast", "player_points_rebounds_assists": "pts+reb+ast",
    "player_threes": "3pm",
}


def api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    env = config.ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("ODDS_API_KEY not set (add ODDS_API_KEY=... to .env at the repo root)")
    if key in {"your_key_here", "..."} or len(key) < 20:
        raise SystemExit("ODDS_API_KEY looks like a placeholder; paste the real key into .env")
    return key


class BudgetExceeded(RuntimeError):
    pass


class NotFound(RuntimeError):
    """404 (e.g. a historical event that was postponed or re-issued). Free; skip it."""


@dataclass
class Budget:
    """Caps credits spent in one run; tracks the account's remaining credits from headers."""

    max_credits: int
    spent: int = 0
    remaining: int | None = None
    log: list[str] = field(default_factory=list)
    reserve: int = RESERVE  # never let the account drop below this

    def check(self, estimate: int, what: str) -> None:
        if self.spent + estimate > self.max_credits:
            raise BudgetExceeded(f"{what}: ~{estimate} credits would exceed run budget "
                                 f"({self.spent}/{self.max_credits} spent)")
        if self.remaining is not None and self.remaining - estimate < self.reserve:
            raise BudgetExceeded(f"{what}: would leave < {self.reserve} credits "
                                 f"(account has {self.remaining})")

    def record(self, headers, what: str) -> None:
        last = int(headers.get("x-requests-last", 0) or 0)
        self.spent += last
        if headers.get("x-requests-remaining") is not None:
            self.remaining = int(float(headers["x-requests-remaining"]))
        self.log.append(f"{what}: cost {last}, account remaining {self.remaining}")


def _get(path: str, params: dict, budget: Budget, estimate: int, what: str,
         session=requests) -> dict | list:
    budget.check(estimate, what)
    for attempt, wait in enumerate((0, 5, 20, 60)):
        if wait:
            time.sleep(wait)
        try:
            resp = session.get(f"{BASE}{path}", params={**params, "apiKey": api_key()}, timeout=60)
            break
        except (requests.Timeout, requests.ConnectionError) as err:
            # never echo the request: its URL contains the key
            print(f"  {what}: {type(err).__name__}, retry {attempt + 1}/3", flush=True)
    else:
        raise RuntimeError(f"{what}: no response after retries")
    budget.record(resp.headers, what)
    if resp.status_code == 404:
        raise NotFound(f"{what}: HTTP 404: {resp.text[:200]}")
    if resp.status_code != 200:
        # never echo the URL: it contains the key
        raise RuntimeError(f"{what}: HTTP {resp.status_code}: {resp.text[:300]}")
    time.sleep(0.3)
    return resp.json()


def _save(kind: str, name: str, payload) -> Path:
    path = RAW_DIR / kind / f"{name}.json"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload))
    tmp.rename(path)
    return path


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def events(budget: Budget, session=requests) -> list[dict]:
    """Upcoming NBA events (free endpoint)."""
    return _get(f"/sports/{SPORT}/events", {}, budget, 0, "events", session)


def game_spreads(budget: Budget, bookmakers: str = AU_BASE, session=requests) -> pl.DataFrame:
    """Point spreads for upcoming games (featured odds endpoint, ~1 credit). Cached under spreads/."""
    from datetime import datetime

    data = _get(f"/sports/{SPORT}/odds", {"bookmakers": bookmakers, "markets": "spreads", "oddsFormat": "decimal"},
                budget, 1, "spreads", session)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RAW_DIR / "spreads" / f"{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    return parse_spreads(data)


def parse_spreads(data: list) -> pl.DataFrame:
    rows = []
    for e in data:
        for bk in e.get("bookmakers", []):
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append({"event_id": e["id"], "commence_time": e["commence_time"], "team_name": o["name"],
                                 "spread": float(o["point"])})
    return pl.DataFrame(rows, schema={"event_id": pl.String, "commence_time": pl.String, "team_name": pl.String,
                                      "spread": pl.Float64}).unique(["event_id", "team_name"])


def latest_spreads() -> pl.DataFrame:
    files = sorted((RAW_DIR / "spreads").glob("*.json"))
    return parse_spreads(json.loads(files[-1].read_text())) if files else parse_spreads([])


def event_markets(event_id: str, regions: str, budget: Budget, session=requests) -> dict:
    """Which markets each bookmaker currently offers for one event (costs ~1)."""
    out = _get(f"/sports/{SPORT}/events/{event_id}/markets", {"regions": regions}, budget, 1,
               f"markets {event_id}", session)
    _save("markets", f"{event_id}_{_stamp()}", out)
    return out


def _book_params(regions: str | None, bookmakers: list[str] | None) -> tuple[dict, int]:
    """API params and the region-equivalent count used for cost (10 bookmakers = 1 region)."""
    if bookmakers:
        return {"bookmakers": ",".join(bookmakers)}, math.ceil(len(bookmakers) / 10)
    return {"regions": regions}, len(regions.split(","))


def event_odds(event_id: str, markets: list[str], regions: str | None, budget: Budget,
               session=requests, bookmakers: list[str] | None = None) -> dict:
    params, n_regions = _book_params(regions, bookmakers)
    out = _get(f"/sports/{SPORT}/events/{event_id}/odds",
               {**params, "markets": ",".join(markets), "oddsFormat": "decimal"},
               budget, len(markets) * n_regions, f"odds {event_id}", session)
    _save("live", f"{event_id}_{_stamp()}", out)
    return out


def historical_events(date_iso: str, budget: Budget, session=requests) -> dict:
    return _get(f"/historical/sports/{SPORT}/events", {"date": date_iso}, budget, 1,
                f"historical events {date_iso}", session)


def historical_event_odds(event_id: str, date_iso: str, markets: list[str], regions: str | None,
                          budget: Budget, session=requests,
                          bookmakers: list[str] | None = None) -> dict:
    params, n_regions = _book_params(regions, bookmakers)
    out = _get(f"/historical/sports/{SPORT}/events/{event_id}/odds",
               {"date": date_iso, **params, "markets": ",".join(markets), "oddsFormat": "decimal"},
               budget, 10 * len(markets) * n_regions, f"historical odds {event_id}", session)
    _save("historical", f"{event_id}_{date_iso.replace(':', '')}", out)
    return out


def backfill(game_days: list, budget: Budget, markets: list[str], bookmakers: list[str],
             hours_before: float = 2.0, seed: int = 0, session=requests,
             shuffle: bool = True) -> int:
    """Historical props for every game on the given ET game days (random order unless shuffle=False).

    Snapshot = tip-off minus hours_before. Resumable: events already cached are skipped (no
    credits). Stops cleanly when the budget or reserve would be breached. Returns games saved.
    """
    import random
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    days = list(game_days)
    if shuffle:
        random.Random(seed).shuffle(days)
    saved = 0
    for d in days:
        done_marker = RAW_DIR / "historical" / "_days" / f"{d.isoformat()}.done"
        if done_marker.exists():
            continue
        noon_utc = datetime(d.year, d.month, d.day, 16, 0, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        evs = historical_events(noon_utc, budget, session).get("data", [])
        todays = [e for e in evs if datetime.fromisoformat(e["commence_time"])
                  .astimezone(et).date() == d]
        for ev in todays:
            tip = datetime.fromisoformat(ev["commence_time"])
            when = (tip - timedelta(hours=hours_before)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if (RAW_DIR / "historical" / f"{ev['id']}_{when.replace(':', '')}.json").exists():
                continue
            try:
                historical_event_odds(ev["id"], when, markets, None, budget, session, bookmakers)
            except NotFound:
                print(f"  skip {ev['away_team']} @ {ev['home_team']} ({ev['commence_time']}): not found",
                      flush=True)
                continue
            saved += 1
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.write_text(str(len(todays)))
        print(f"{d}: {len(todays)} games | spent {budget.spent} | remaining {budget.remaining}",
              flush=True)
    return saved


def threshold(point: float | None, name: str) -> int | None:
    """'Over 19.5' -> 20 (i.e. 20+). Milestone lines without a point are already X+."""
    if point is None:
        return None
    return math.floor(point) + 1 if name.lower() == "over" else None


def parse_event_odds(payload: dict, snapshot: str) -> pl.DataFrame:
    """Tidy rows: one per bookmaker x market x player x line. Handles live and historical shape."""
    ev = payload.get("data", payload)
    rows = []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            base = mk["key"].removesuffix("_alternate")
            for o in mk.get("outcomes", []):
                rows.append({
                    "snapshot": snapshot, "event_id": ev["id"], "commence_time": ev["commence_time"],
                    "home_team": ev["home_team"], "away_team": ev["away_team"],
                    "bookmaker": bk["key"], "market": mk["key"],
                    "stat": STAT_OF_MARKET.get(base, base), "alternate": mk["key"].endswith("_alternate"),
                    "player": o.get("description"), "side": o.get("name"), "point": o.get("point"),
                    "threshold": threshold(o.get("point"), o.get("name", "")),
                    "price": o.get("price"), "last_update": mk.get("last_update"),
                })
    schema = {"snapshot": pl.String, "event_id": pl.String, "commence_time": pl.String,
              "home_team": pl.String, "away_team": pl.String, "bookmaker": pl.String,
              "market": pl.String, "stat": pl.String, "alternate": pl.Boolean, "player": pl.String,
              "side": pl.String, "point": pl.Float64, "threshold": pl.Int64, "price": pl.Float64,
              "last_update": pl.String}
    return pl.DataFrame(rows, schema=schema)


def load_cached(kind: str = "live") -> pl.DataFrame:
    """All cached responses of a kind, parsed (no network)."""
    frames = []
    for p in sorted((RAW_DIR / kind).glob("*.json")):
        snap = p.stem.split("_", 1)[1] if "_" in p.stem else p.stem
        frames.append(parse_event_odds(json.loads(p.read_text()), snap))
    return pl.concat(frames) if frames else parse_event_odds({"bookmakers": [], "id": ""}, "")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage", help="which books offer which NBA markets (~1 credit)")
    c.add_argument("--regions", default="us,au")
    sub.add_parser("spreads", help="point spreads for upcoming games (~1 credit)")
    s = sub.add_parser("snapshot", help="player props for upcoming games")
    s.add_argument("--regions", default=None, help="instead of --bookmakers, e.g. us,au")
    s.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS))
    s.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    s.add_argument("--max-credits", type=int, default=100)  # 9+ games x 7 credits: 40 silently dropped games 6-9
    s.add_argument("--max-events", type=int, default=20)
    s.add_argument("--within-hours", type=float, default=36.0, help="every game tipping within this window")
    h = sub.add_parser("historical", help="props as of a past time (paid plans only)")
    h.add_argument("--date", required=True, help="ISO time, e.g. 2026-03-15T23:00:00Z")
    h.add_argument("--regions", default=None)
    h.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS))
    h.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    h.add_argument("--max-credits", type=int, default=100)
    b = sub.add_parser("backfill", help="historical props for a whole season (resumable)")
    b.add_argument("--season-id", default="22025")
    b.add_argument("--markets", default=",".join(DEFAULT_MARKETS[:4]))  # pts, reb, ast, P+R ladders
    b.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS))
    b.add_argument("--max-credits", type=int, required=True)
    b.add_argument("--reserve", type=int, default=RESERVE, help="stop if the account would drop below this")
    b.add_argument("--prioritize-spots", action="store_true",
                   help="days with the most backup-big spots (starter out) first, instead of random")
    args = ap.parse_args()

    budget = Budget(max_credits=getattr(args, "max_credits", 5), reserve=getattr(args, "reserve", RESERVE))
    try:
        if args.cmd == "coverage":
            evs = events(budget)
            if not evs:
                raise SystemExit("no upcoming NBA events")
            mk = event_markets(evs[0]["id"], args.regions, budget)
            print(f"{evs[0]['away_team']} @ {evs[0]['home_team']} ({evs[0]['commence_time']}):")
            for bk in mk.get("bookmakers", []):
                keys = sorted(m["key"] for m in bk.get("markets", []))
                props = [k for k in keys if k.startswith("player_")]
                print(f"  {bk['key']:22s} {len(keys):3d} markets, {len(props):3d} player props: "
                      f"{', '.join(props[:8])}{' ...' if len(props) > 8 else ''}")
        elif args.cmd == "spreads":
            sp = game_spreads(budget)
            print(sp.sort("commence_time"))
        elif args.cmd == "snapshot":
            markets = args.markets.split(",")
            from datetime import datetime, timedelta

            horizon = (datetime.now(UTC) + timedelta(hours=args.within_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
            evs = [e for e in events(budget) if e["commence_time"] <= horizon][: args.max_events]
            frames = []
            for ev in evs:
                books = None if args.regions else args.bookmakers.split(",")
                payload = event_odds(ev["id"], markets, args.regions, budget, bookmakers=books)
                frames.append(parse_event_odds(payload, _stamp()))
            df = pl.concat(frames) if frames else pl.DataFrame()
            print(f"{len(evs)} events, {df.height:,} prices, "
                  f"{df['player'].n_unique() if df.height else 0} players")
        elif args.cmd == "backfill":
            pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
            days = pg.filter(pl.col("season_id") == args.season_id)["game_date"].unique().sort().to_list()
            shuffle = True
            if args.prioritize_spots:
                bg = pl.read_parquet(config.DATA_PROCESSED / "backup_big_games.parquet")
                score = (bg.filter(pl.col("season_id") == args.season_id, pl.col("starter_out"))
                         .group_by("game_date").len())
                ranked = score.sort("len", descending=True)["game_date"].to_list()
                days = ranked + [d for d in days if d not in set(ranked)]
                shuffle = False
            n = backfill(days, budget, args.markets.split(","), args.bookmakers.split(","),
                         shuffle=shuffle)
            print(f"backfill: {n} games saved this run")
        elif args.cmd == "historical":
            markets = args.markets.split(",")
            evs = historical_events(args.date, budget).get("data", [])
            for ev in evs:
                books = None if args.regions else args.bookmakers.split(",")
                historical_event_odds(ev["id"], args.date, markets, args.regions, budget,
                                      bookmakers=books)
            print(f"{len(evs)} historical events saved")
    except BudgetExceeded as err:
        print(f"STOPPED: {err}")
    finally:
        print(f"credits spent this run: {budget.spent}; account remaining: {budget.remaining}")
        for line in budget.log[-5:]:
            print("  " + line)


if __name__ == "__main__":
    main()
