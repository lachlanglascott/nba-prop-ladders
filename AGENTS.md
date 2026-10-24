# nba-prop-edge

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

## Modelling principles
1. Model **minutes first** as a mixture over named scenarios; then per-minute rates; then
   simulate the joint outcome. Report P(A and B) next to P(A)·P(B) — the gap is the edge.
2. Every prediction must be reproducible "as-at" a timestamp (only info available before
   tip-off — injury report time matters).
3. Evaluate with calibration + closing-line value, not just hit rate.

## Layout
- `src/nba_edge/` — package code (`config.py` holds paths)
- `tests/` — pytest; keep tests offline (no network) by using fixtures
- `bets/bet_log.csv` — every real or paper bet, incl. losers. Use the `log-bet` skill.
- `docs/` — notes, decisions, examples

## Commands
- Install: `uv sync`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check .`

## Conventions
- Python 3.11, type hints, small pure functions; network I/O isolated in `src/nba_edge/sources/`.
- Prefer polars/duckdb for tabular work, parquet for storage.
