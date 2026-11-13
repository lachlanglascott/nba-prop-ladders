# nba-prop-ladders

A personal model for finding +EV NBA player props and same-game multis (SGMs). It focuses on role and
backup players, whose minutes depend on who else is playing tonight.

It models the **joint** distribution of points, rebounds and assists. Minutes are treated as a mixture of
named scenarios (starter out, foul trouble, blowout, hot hand), not one normal curve. Each prediction is
then checked against the market, not just against hit rate. Public data only: nba_api / stats.nba.com,
official NBA injury reports, and The Odds API.

## How a projection is built

1. **Minutes first.** Start from his games that match tonight's lineup, this season and last. Absent
   teammates free minutes (pair effects, shrunk to a position prior). The point spread sets the blowout risk,
   with separate minute curves for the winning and losing sides.
2. **Per-minute rates.**
   - **Points:** points per 36.
   - **Assists:** potential assists × his career conversion.
   - **Rebounds:** rebound chances × his career conversion.
   - **Usage** (`usage.py`): when teammates sit, each player's own past games without them, in any season
     and lineup, shrunk to a league prior.
   - **Opposition by position** (`opp_position.py`): what tonight's opponent allows to guards / wings / bigs,
     for every stat, as at the game date.
3. **Simulate** the joint outcome (40k draws). Price singles and P/R, R/A and P/R/A combos, and report
   P(A and B) next to P(A)·P(B). The gap between them is the SGM edge.
4. **Honest fair price.** Start from the market's own probability and move toward the model only by a
   confidence (0.08–0.35) set by the spot type and sample size. Bets need positive EV on that number.

## Angles (hypotheses, each tested separately)

Full table and findings: [`docs/angles.md`](docs/angles.md).

- **Starter-out backup bigs.** The headline alert: starting centre Out or Doubtful, so his backup's P/R ladder.
- **Role change, conversion bounce, assist luck reversion, all-round P/R/A.**
- **Angle H, the opposition spot** ([`docs/opp_position.md`](docs/opp_position.md)). Sportsbet ignores most
  of what an opponent allows to a position group. Bigs' P/R bets in a bad spot are passed; good spots are boosted.

Single legs at AU books are close to efficient. The edge so far comes from lineup context, correlation in
SGMs, and stacking angles.

## Setup

```bash
uv sync                      # Python 3.11, pinned in .python-version
echo "ODDS_API_KEY=..." > .env   # only for odds snapshots
uv run pytest -q             # offline tests
uv run ruff check .
```

`data/raw/` is an append-only cache of every source response and `data/processed/` holds derived tables.
Both are git-ignored, so pull the data first (below).

## Daily routine (US game dates)

```bash
uv run python -m nba_ladders.pull --seasons 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26 2026-27 --refresh 2026-27
uv run python -m nba_ladders.spots --date <yesterday> --grade      # grade last night's slate
uv run python -m nba_ladders.sources.odds_api spreads              # blowout risk
uv run python -m nba_ladders.sources.odds_api snapshot             # prices (credit-budgeted)
uv run python -m nba_ladders.spots --date <today>                  # → docs/slates/<date>-spots.{md,html}
```

Re-run `spots` after the 5 PM ET injury report, and use `--out/--start/--bench` for late news. Every real
or paper bet goes in `bets/bet_log.csv`.

## Useful tools

| Command | What it does |
|---|---|
| `python -m nba_ladders.price "Player" [--starter-out] [--opponent XXX]` | Fair prices for one player's ladder |
| `python -m nba_ladders.ppm "Player"` | Per-minute rates and which teammates' absences unlock minutes |
| `python -m nba_ladders.usage --player "Player" --out "A,B" --date D` | Usage bump tonight, per absent teammate |
| `python -m nba_ladders.opp_position [--player X]` | Opposition-by-position tests: predictive and priced |
| `python -m nba_ladders.combos "Player" --opponent XXX` | Best P/R/A combos by correlation lift |
| `python -m nba_ladders.luck --scan` | Potential-assist / rebound-chance luck scan |

Prefix each with `uv run`. Full list: [`AGENTS.md`](AGENTS.md). What every file is for: [`docs/REPO_GUIDE.md`](docs/REPO_GUIDE.md).

## Layout

```
src/nba_ladders/          package (network I/O only in sources/)
tests/                    pytest, offline fixtures
docs/                     findings, angle write-ups, slates and assessments (docs/slates/)
bets/bet_log.csv          every bet, losers included
```

## Rules

- Public sources only. Respect rate limits (stats.nba.com ≥ 0.6 s between calls, everything cached).
- No bookmaker scraping.
- Every prediction is reproducible as at a timestamp, using only information available before tip-off.
- Judge the model on calibration and closing-line value, not hit rate.

Personal project, for analysis only, not financial advice. Gambling Help 1800 858 858 · BetStop betstop.gov.au
