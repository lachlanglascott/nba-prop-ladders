# nba-prop-ladders

Personal project. Find +EV NBA player-prop and same-game-multi (SGM) bets by modelling
the **joint** distribution of correlated stats (PTS/REB/AST) for role/backup players,
where minutes are a *scenario mixture* (starter out, foul trouble, blowout, hot hand),
not a normal distribution.

## Hard rules
- **No employer data, systems, or knowledge.** Public sources only (nba_api, Basketball-Reference,
  official NBA injury reports, public odds APIs). Never reference internal pricing logic.
- `data/raw/` is append-only cache of source pulls — never edit or delete files there.
  Derived data goes in `data/processed/`. Both are git-ignored.
- Respect source rate limits (stats.nba.com: >=0.6s between calls, cache every response).
- Don't scrape bookmaker sites without the user's explicit go-ahead (ToS / account risk).

## Priorities
- **Headline alert: starting centre Out/Doubtful → his backup big's P/R (and P/R/A) ladder.**
  This is the user's core skill; it always goes first in sheets and notifications.
- Then everyone else: any player, any teammate-out combination, 2- and 3-leg combos.

## Modelling principles
1. Model **minutes first** as a mixture over named scenarios; then per-minute rates; then
   simulate the joint outcome. Report P(A and B) next to P(A)·P(B) — the gap is the edge.
2. Every prediction must be reproducible "as-at" a timestamp (only info available before
   tip-off — injury report time matters).
3. Evaluate with calibration + closing-line value, not just hit rate.

## Layout
- `src/nba_ladders/` — package code (`config.py` holds paths)
- `tests/` — pytest; keep tests offline (no network) by using fixtures
- `bets/bet_log.csv` — every real or paper bet, incl. losers. Use the `log-bet` skill.
- `docs/` — notes, decisions, examples

## Commands
- Pull data: `uv run python -m nba_ladders.pull --seasons 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26 2026-27 --refresh 2026-27`
- Backup-bigs screen: `uv run python -m nba_ladders.reports` (writes `docs/screens/`)
- Angles (hypotheses being tested): `docs/angles.md`
- Rebuild scenarios: `uv run python -m nba_ladders.scenarios`; backtest: `uv run python -m nba_ladders.backtest`
- Game-day sheet: `uv run python -m nba_ladders.sheet [--date YYYY-MM-DD]` (official injury report → alerts)
- Per-minute report: `uv run python -m nba_ladders.ppm "Player Name"` (teammate unlocks; see `docs/context_model.md`)
- Status calibration: `uv run python -m nba_ladders.status_calibration --season-id 22025`
- Tracking (potential assists, rebound chances): `uv run python -m nba_ladders.sources.tracking --seasons 2024-25 2025-26`; luck test `uv run python -m nba_ladders.luck ["Player"]`
- Combo explorer: `uv run python -m nba_ladders.combos "Player" --opponent XXX [--as-of YYYY-MM-DD]` (best P/R/A combos by correlation lift)
- Potentials scan: `uv run python -m nba_ladders.luck --scan [YYYY-MM-DD]`
- Odds (The Odds API, key in `.env` as ODDS_API_KEY): `uv run python -m nba_ladders.sources.odds_api coverage|snapshot|historical` (hard credit budget per run; never print the key)
- Market vs model: `uv run python -m nba_ladders.market [--player X] [--all]` (needs a snapshot; flags are leads, not bets, until backtested)
- **Game-day write-up / candidates**: `uv run python -m nba_ladders.candidates [--date YYYY-MM-DD]` → `docs/slates/<date>.md` (take an odds snapshot first)
- Calibrated blend (book + model, fit on the market backtest): `uv run python -m nba_ladders.blend fit`
- Price a ladder: `uv run python -m nba_ladders.price "Player Name" [--starter-out] [--opponent MEM] [--combos 10p/10r,12r]` (see `docs/simulator.md`)
- Install: `uv sync`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check .`

## Conventions
- Python 3.11, type hints, small pure functions; network I/O isolated in `src/nba_ladders/sources/`.
- Prefer polars/duckdb for tabular work, parquet for storage.
- Python pinned to 3.11 (`.python-version`). HTTPS uses `truststore` (OS cert store) so requests work behind TLS-inspecting proxies — don't disable cert verification.
