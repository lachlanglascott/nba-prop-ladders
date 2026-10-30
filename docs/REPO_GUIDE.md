# Repo guide

What every file in this repo is for, grouped by job.

## Instructions for the agent

| File | Purpose |
|---|---|
| `AGENTS.md` | **The project brief.** Goal, hard rules (public data only, never edit raw data, rate limits, no bookmaker scraping without explicit OK), modelling principles, folder layout, commands. Loaded at the start of every session. If you correct the agent twice about the same thing, add it here. |
| `CLAUDE.md` | One line, `@AGENTS.md`, which tells Claude Code to read that file. Codex reads `AGENTS.md` directly, so one brief serves both tools. Claude-only notes go below that line. |

## Claude Code configuration (`.claude/`)

| File | Purpose |
|---|---|
| `.claude/settings.json` | Project permissions and hooks: commands that don't need approval (`pytest`, `ruff`, `uv sync`, `git status/diff`), a block on editing `data/raw/`, and a hook that runs the tests after every edit and sends failures back to the agent. Committed, so it travels with the repo. Hooks take effect from a new session. |
| `.claude/skills/log-bet/SKILL.md` | **A skill**: a playbook that only loads when its description matches the task ("record this bet"). It says which fields to collect, the allowed scenario labels, what to ask for, and never to invent stats. Skills keep specialised procedures out of `AGENTS.md` so they don't use up context every session. |
| `docs/claude-settings.example.json` | Reference copy of the settings above (written when auto mode blocked the agent from editing settings directly). Can be deleted. |

## Python project

| File | Purpose |
|---|---|
| `pyproject.toml` | The project definition: name, Python version, packages it needs (`nba_api` for NBA stats, `polars`/`duckdb` for tables, `numpy`/`scipy` for simulation), dev tools (`pytest`, `ruff`) and lint settings. `uv add <pkg>` updates it. |
| `.python-version` | Pins Python 3.11 for uv. |
| `uv.lock` | Exact version of every package, including ones pulled in indirectly. `uv sync` reproduces the same environment on any machine. Never edit by hand. |
| `src/nba_ladders/__init__.py` | Empty. Marks the folder as a Python package so `from nba_ladders import config` works. |
| `src/nba_ladders/sources/nba_stats.py` | Cached, throttled stats.nba.com fetcher. Each raw response saved once as `data/raw/nba_stats/leaguegamelog/<season>_<type>_<date>.json`; completed seasons fetched once, the current season gets a new dated snapshot when refreshed. Never overwrites. |
| `src/nba_ladders/player_games.py` | Turns the raw response into a tidy player-game table (minutes, pts, reb, ast, fouls, home/away, opponent). |
| `src/nba_ladders/with_without.py` | For every pair of teammates: a player's minutes/stats when the teammate played vs was absent (only counting games the teammate was on the roster, so trades don't count as absences). Finds "backup gets minutes when X sits" pairs. |
| `src/nba_ladders/pull.py` | Command that pulls the seasons and writes `data/processed/player_games.parquet`. |
| `src/nba_ladders/screens.py` | Candidate screens. `backup_bigs` = current bigs in a bench role last season, with minutes spread, production at 25+ min, blocker last season and centre ahead now. |
| `src/nba_ladders/reports.py` | Writes screen outputs to `docs/screens/<name>_<date>.csv`. |
| `src/nba_ladders/scenarios.py` | Pre-game scenarios: starting centre per team-season, backup-big games flagged `starter_out`. `python -m nba_ladders.scenarios` rebuilds the table. |
| `src/nba_ladders/angles_abc.py` | Season-to-season predictability tests for angles A–C. |
| `src/nba_ladders/sim.py` | Monte Carlo simulator for joint pts/reb; fair price per ladder rung. |
| `src/nba_ladders/backtest.py` | As-at backtest of the simulator with calibration and scores. |
| `src/nba_ladders/price.py` | CLI: fair prices for a player's ladder in a scenario. |
| `src/nba_ladders/sources/pbp.py` | Cached play-by-play fetcher (one JSON per game, never re-fetched), foul and starter parsing. `python -m nba_ladders.sources.pbp --season-id 22025`. |
| `src/nba_ladders/fouls.py` | Angle D analysis: starting centres' early foul trouble, backup uplift, opponent effect, persistence. |
| `src/nba_ladders/sources/injury_report.py` | Official NBA injury report (15-min PDFs): fetch, cache, parse to player/status/reason; name-matching keys. |
| `src/nba_ladders/sheet.py` | Game-day sheet: starting centres Out/Doubtful (alerts, new absences first) and Questionable (watch) → backups' fair P/R and P/R/A prices. |
| `src/nba_ladders/absences.py` | Teammate-absence effects on minutes and PPM/RPM/APM: pair with/without, league prior by position, shrinkage, combining several absences. |
| `src/nba_ladders/context_sim.py` | Context-aware simulator: un-adjust past games for who sat, re-adjust for the target game. |
| `src/nba_ladders/backtest_context.py` | As-at backtest of the context model vs the no-context model, all players. |
| `src/nba_ladders/ppm.py` | Per-minute report: full-strength minutes, PPM/RPM/APM, and which teammates unlock a player. |
| `src/nba_ladders/sources/tracking.py` | Per-game tracking stats (potential/secondary assists, passes, rebound chances, contested boards) via one-day LeagueDashPtStats queries, cached. |
| `src/nba_ladders/luck.py` | Opportunity vs outcome: do potential assists / rebound chances predict future production better than actuals? Per-player luck report. |
| `src/nba_ladders/status_calibration.py` | P(sits) by injury-report status, from past reports vs box scores. |
| `src/nba_ladders/rebounding.py` | Rebounding vs opponent and by position: expected rebounds (as-at), opponent factors, persistence tests. |
| `src/nba_ladders/calibration.py` | Logistic recalibration of simulated probabilities (fit one season, check the next). Tried; not applied. |
| `src/nba_ladders/combos.py` | Combo explorer: grid of P/R, P/A, R/A, P/R/A combos with fair price, independent-legs price and correlation lift; best per shape. |
| `src/nba_ladders/sources/odds_api.py` | The Odds API v4 loader: coverage / live snapshot / historical player props, credit budget + reserve, raw cache, parser to X+ thresholds. |
| `src/nba_ladders/config.py` | One place for settings: data paths, bet log location, stats.nba.com throttle (0.6s between calls), allowed scenario labels. Code imports from here instead of hard-coding paths. |

## Tests

| File | Purpose |
|---|---|
| `tests/test_phase1.py` | Offline tests for the parser, the never-overwrite cache rule, and with/without logic (on a hand-built 5-game team with a starter, backup and traded player). |
| `tests/test_sim.py` | Simulator tests: bounds, starter-out lifts minutes, correlated rates make joint > independent. |
| `tests/test_fouls.py` | Offline tests for foul parsing, starter inference and the foul-trouble analysis. |
| `tests/test_injury_report.py`, `tests/test_sheet.py` | Offline tests for report parsing, name keys, depth chart and new-absence logic. |
| `tests/test_absences.py`, `tests/test_luck.py`, `tests/test_status_calibration.py` | Offline tests for absence effects, as-at luck features, status matching. |
| `tests/test_combos.py` | Combo explorer: lift > 1 when legs share minutes, price range, realistic legs, per-shape picks. |
| `tests/test_odds_api.py` | Offline loader tests (fake HTTP): thresholds, historical wrapper, budget refusal and reserve, caching. |
| `tests/test_smoke.py` | Quick checks: data folders exist, bet log has the right columns, every scenario label is a known one. The hook runs these after each edit. Data-pipeline tests will use saved sample data so tests never need the network. |

## Data and bets

| File | Purpose |
|---|---|
| `bets/bet_log.csv` | **Labelled bet history**: every real or paper bet, losers included, with scenario and reasoning. This is the validation set (e.g. did "new starter" cases ever price well?). |
| `data/raw/.gitkeep`, `data/processed/.gitkeep` | Git doesn't track empty folders, so these empty files keep the folders in the repo. The data inside is ignored and never reaches GitHub. `raw/` = untouched cached source responses; `processed/` = cleaned tables built from them. |
| `.gitignore` | What git skips: the environment, tool caches, data files, `.env` (secrets like API keys), local Claude settings, `.DS_Store`. |

## Docs

| File | Purpose |
|---|---|
| `docs/angles.md` | The betting angles as separate hypotheses (conversion, tail, injury trigger, foul trouble): metric, market, status, findings. |
| `docs/screens/` | Dated screen outputs (CSV) for review. |
| `docs/angles_abc_findings.md` | Results for angles A–C. |
| `docs/simulator.md` | How the simulator works, backtest and calibration results, limitations. |
| `docs/angle_d_fouls.md` | Results for angle D (foul trouble). |
| `docs/context_model.md` | Any-teammate-out model: method, tuning, holdout results, next steps. |
| `docs/rebounding.md` | Opponent and position rebounding effects; Camara worked example. |
| `docs/combos.md` | Combo explorer: why (Achiuwa 24p/9r), how lift works, replay. |
| `docs/luck.md` | Potential assists / rebound chances: does luck revert? Blend weights. |
| `docs/status_calibration.md` | Injury-report status → P(sits), 2025-26. |
| `docs/phase1_findings.md` | First results from the data. |
| `docs/REPO_GUIDE.md` | This file. Update it when files are added or their role changes. |

## Local only (not in git)

| Path | Purpose |
|---|---|
| `.venv/` | Virtual environment with installed packages. Rebuilt by `uv sync`, never committed. |
| `.pytest_cache/`, `.ruff_cache/` | Speed-up caches for the test and lint tools. Safe to delete. |
| `.git/` | The git history itself. |

## Everyday commands

```bash
uv sync              # install / update the environment from uv.lock
uv add <package>     # add a dependency (updates pyproject.toml + uv.lock)
uv run pytest -q     # run tests
uv run ruff check .  # lint
```
