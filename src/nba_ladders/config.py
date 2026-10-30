from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
BET_LOG = ROOT / "bets" / "bet_log.csv"

# stats.nba.com throttles aggressively; keep at least this gap between calls.
NBA_API_MIN_SECONDS_BETWEEN_CALLS = 0.6

# Why we expected the minutes/production. Keep in sync with the log-bet skill.
SCENARIOS = frozenset(
    {"starter_out", "new_starter", "role_change", "luck_reversion", "foul_trouble_opp", "blowout_risk",
     "matchup", "other"}
)
