---
name: log-bet
description: Use when the user describes a bet they placed or are considering (player, legs, odds, context) and wants it recorded, or asks to update a bet's result. Appends/updates rows in bets/bet_log.csv.
---

# Log a bet

1. Extract fields for `bets/bet_log.csv` (keep column order exactly as the header):
   date (YYYY-MM-DD), player, team, opponent, bet_type (single / SGM / SGM ladder),
   legs (e.g. "PTS 12+ & REB 8+"), odds (decimal), stake, scenario, thesis, result (W/L/P/open),
   actual_line (e.g. "9 pts / 7 reb / 24 min"), notes.
2. `scenario` must be one of: `starter_out`, `new_starter`, `foul_trouble_opp`, `blowout_risk`,
   `matchup`, `other` — ask if unclear. This is the label the model will learn from.
3. Ask only for fields that matter and are missing: **odds, game date, and why minutes were expected**.
   Leave others blank rather than guessing. Never invent stats — if actuals are needed, say they'll be
   back-filled from nba_api once the data pipeline exists.
4. Append (or update the matching row), then run `uv run pytest -q` to confirm the CSV still parses.
5. One-line summary back to the user: what was logged and what's still blank.
