"""Official NBA injury report (PDF, published every 15 minutes on game days).

    https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH>_<MM><AM|PM>.pdf

Times are US Eastern. Each PDF is cached once at data/raw/injury_reports/ and never overwritten,
which also gives a timestamped history for as-at backtests ("what did we know before tip").
"""

import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pdfplumber
import polars as pl
import requests
import truststore

from nba_ladders import config

truststore.inject_into_ssl()

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
RAW_DIR = config.DATA_RAW / "injury_reports"
ET = ZoneInfo("America/New_York")

# Left edge (x0, points) where each column starts on the landscape page.
COLUMNS = [
    ("game_date", 0), ("game_time", 115), ("matchup", 195), ("team_name", 260),
    ("player", 420), ("status", 580), ("reason", 660),
]
STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
TITLE_BAND = 70  # page title "Injury Report: <date> <time>" sits above this on every page
NOT_SUBMITTED = "NOTYETSUBMITTED"

SCHEMA = {
    "report_time": pl.Datetime, "game_date": pl.Date, "game_time": pl.String,
    "matchup": pl.String, "team_name": pl.String, "player": pl.String,
    "player_key": pl.String, "status": pl.String, "reason": pl.String,
}


def stamp(ts: datetime) -> str:
    """2026-10-24 18:00 ET -> '2026-10-24_06_00PM'."""
    return ts.strftime("%Y-%m-%d_%I_%M%p")


def report_path(ts: datetime, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"Injury-Report_{stamp(ts)}.pdf"


def fetch_report(ts: datetime, raw_dir: Path = RAW_DIR) -> Path | None:
    """Download the report published at ts (ET) if it exists; cached, never overwritten."""
    path = report_path(ts, raw_dir)
    if path.exists():
        return path
    resp = requests.get(URL.format(stamp=stamp(ts)), timeout=30)
    if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
        return None  # S3 answers 403 AccessDenied for reports that don't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pdf.part")
    tmp.write_bytes(resp.content)
    tmp.rename(path)
    return path


def latest_report(now: datetime | None = None, lookback_hours: int = 12,
                  raw_dir: Path = RAW_DIR) -> tuple[datetime, Path] | None:
    """Most recent published report at or before now (ET), stepping back 15 minutes."""
    now = (now or datetime.now(ET)).astimezone(ET)
    ts = now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)
    for _ in range(lookback_hours * 4):
        if path := fetch_report(ts, raw_dir):
            return ts, path
        ts -= timedelta(minutes=15)
    return None


def player_key(name: str) -> str:
    """Match key for 'Vučević, Nikola', 'Nikola Vucevic', 'Udeh,Jr.,Ernest', 'Ernest Udeh Jr.'."""
    parts = [p.strip() for p in name.split(",")]
    if len(parts) > 1:  # report style "Last,[Suffix,]First"
        name = " ".join([parts[-1], *parts[:-1]])
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    tokens = [t for t in re.split(r"[^a-z]+", folded) if t and t not in {"jr", "sr", "ii", "iii", "iv"}]
    return "".join(sorted(tokens))


def _column(x0: float) -> str:
    return [name for name, left in COLUMNS if x0 >= left][-1]


def parse_words(words: list[dict]) -> list[dict]:
    """Turn one page's words (pdfplumber extract_words) into player rows.

    Date/time/matchup/team are only printed on the first row of their block, so they are
    carried down. Long reasons wrap onto lines just above/below the player's own line, so each
    reason fragment is attached to the nearest player line.
    """
    cells = sorted(
        ({"col": _column(w["x0"]), "top": w["top"], "x0": w["x0"], "text": w["text"]} for w in words),
        key=lambda c: (round(c["top"]), c["x0"]),
    )
    header = next((c["top"] for c in cells if c["text"] == "PlayerName"), TITLE_BAND)
    cells = [
        c for c in cells
        if c["top"] > max(header + 2, TITLE_BAND) and not re.match(r"Page\d+of\d+", c["text"])
    ]

    rows: list[dict] = []
    carry = {"game_date": None, "game_time": None, "matchup": None, "team_name": None}
    for c in sorted(cells, key=lambda c: (c["top"], c["x0"])):
        if c["col"] in carry:
            carry[c["col"]] = c["text"]
            if c["col"] == "team_name" and c["text"] != carry.get("_team_seen"):
                carry["_team_seen"] = c["text"]
        elif c["col"] == "player" or c["text"] == NOT_SUBMITTED:
            # A team that hasn't filed gets one row with no player.
            not_filed = c["text"] == NOT_SUBMITTED
            rows.append({**{k: v for k, v in carry.items() if not k.startswith("_")},
                         "player": None if not_filed else c["text"], "top": c["top"],
                         "status": "Not submitted" if not_filed else None, "reason": []})
    same_line = lambda r, c: abs(r["top"] - c["top"]) < 2  # noqa: E731
    reasons = [c for c in cells if c["col"] == "reason" and c["text"] != NOT_SUBMITTED]
    for r in rows:
        r["has_own_reason"] = any(same_line(r, c) for c in reasons)
    for c in cells:
        if not rows or c["col"] not in {"status", "reason"} or c["text"] == NOT_SUBMITTED:
            continue
        # Ties between two player lines go to the one without a reason on its own line:
        # a wrapped reason belongs to the player whose line it surrounds.
        nearest = min(rows, key=lambda r: (abs(r["top"] - c["top"]),
                                           r["has_own_reason"] and not same_line(r, c)))
        if c["col"] == "status" and c["text"] in STATUSES:
            nearest["status"] = c["text"]
        elif c["col"] == "reason":
            nearest["reason"].append((c["top"], c["text"]))
    for r in rows:
        r["reason"] = " ".join(t for _, t in sorted(r["reason"]))
        del r["top"], r["has_own_reason"]
    return rows


def parse_report(path: Path, report_time: datetime) -> pl.DataFrame:
    rows: list[dict] = []
    with pdfplumber.open(path) as pdf:
        carry: dict = {}
        for page in pdf.pages:
            page_rows = parse_words(page.extract_words())
            for r in page_rows:  # blocks continue across page breaks
                for k in ("game_date", "game_time", "matchup", "team_name"):
                    r[k] = r[k] or carry.get(k)
                carry = {k: r[k] for k in ("game_date", "game_time", "matchup", "team_name")}
            rows += page_rows
    df = pl.DataFrame(
        [{**r, "report_time": report_time.replace(tzinfo=None),
          "player_key": player_key(r["player"]) if r["player"] else None}
         for r in rows],
        schema=SCHEMA if not rows else None,
    )
    if df.is_empty():
        return pl.DataFrame(schema=SCHEMA)
    return df.with_columns(pl.col("game_date").str.to_date("%m/%d/%Y")).select(list(SCHEMA))


def report_for_date(d: date) -> pl.DataFrame:
    """Latest report published on game day d (ET), parsed."""
    end = datetime(d.year, d.month, d.day, 23, 45, tzinfo=ET)
    found = latest_report(min(end, datetime.now(ET)), lookback_hours=24)
    return parse_report(found[1], found[0]) if found else pl.DataFrame(schema=SCHEMA)
