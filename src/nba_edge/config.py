from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
BET_LOG = ROOT / "bets" / "bet_log.csv"

# stats.nba.com throttles aggressively; keep at least this gap between calls.
NBA_API_MIN_SECONDS_BETWEEN_CALLS = 0.6
