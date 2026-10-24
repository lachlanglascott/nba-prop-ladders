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
| `uv.lock` | Exact version of every package, including ones pulled in indirectly. `uv sync` reproduces the same environment on any machine. Never edit by hand. |
| `src/nba_ladders/__init__.py` | Empty. Marks the folder as a Python package so `from nba_ladders import config` works. |
| `src/nba_ladders/config.py` | One place for settings: data paths, bet log location, stats.nba.com throttle (0.6s between calls), allowed scenario labels. Code imports from here instead of hard-coding paths. |

## Tests

| File | Purpose |
|---|---|
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
