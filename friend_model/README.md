# Friend's model (port) — kept separate from ours

Python port of "NBA Alt-Ladder & SGM Model: Complete Handoff" (11 pages, received 4 Nov 2026).
Same formulas, weights and rules as the handoff's Section 4 and its JavaScript (5.4 model + trend
screen, 5.6 price screen), with legitimate data sources instead of browser scraping:

| Handoff source | Here |
|---|---|
| nba.com stats JS (LastNGames windows, tracking, opponent by position, pace) | nba_api game logs + tracking already cached in `data/processed` |
| ESPN injuries page | official NBA injury report (Out/Doubtful = OUT, Questionable = Q) |
| PointsBet spreads API | The Odds API spreads (Sportsbet) |
| Sportsbet page scraper | The Odds API Sportsbet alternate ladders (incl. threes) |

No code from `src/nba_ladders` model modules is used (only the data loaders).

    uv run python friend_model/run.py --date 2026-11-04            # card for a US date
    uv run python friend_model/run.py --date 2026-11-01 --grade    # replay + grade a past date
