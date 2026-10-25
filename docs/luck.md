# Opportunity vs outcome: potential assists and rebound chances (2026-10-26)

Code: `src/nba_ladders/sources/tracking.py` (fetch), `src/nba_ladders/luck.py` (analysis).

```bash
uv run python -m nba_ladders.sources.tracking --seasons 2024-25 2025-26   # re-run fills gaps
uv run python -m nba_ladders.luck                  # predictive test
uv run python -m nba_ladders.luck "Deni Avdija"    # a player's recent luck
```

## Data
Per-game tracking from one-day `LeagueDashPtStats` queries (GP = 1 rows): potential, secondary
and FT assists, passes, rebound chances (off/def), contested/uncontested rebounds. Validated on
2026-03-15: assists match the box score 100%; contested + uncontested = box rebounds 100%.
2023-24 → 2025-26: 79,071 player-games (2 day-measures missing in 2023-24).
League conversion: **56% of potential assists** become assists; **61% of rebound chances**
become rebounds.

## Is "8 potential, only 2 actual" bad luck?

Predict next-game assists/rebounds from the last 10 games' per-minute rate, at the next game's
actual minutes. "Expected" = chances per minute × the player's long-run conversion (shrunk to
league). 2024-25 + 2025-26, ~36k player-games.

By luck over the last 10 games (assists):

| Last 10 games | Next game actual | From recent actual | From potential assists |
|---|---|---|---|
| Very unlucky | 2.23 | 1.82 (−18%) | 2.48 |
| Unlucky | 3.22 | 3.00 | 3.34 |
| Neutral | 3.49 | 3.47 | 3.52 |
| Lucky | 3.27 | 3.53 | 3.29 |
| Very lucky | 2.38 | 2.70 (+13%) | 2.17 |

**Luck partly reverts.** The truth sits about halfway between recent actuals and the
potential-assist expectation. Rebounds show the same pattern (very unlucky: 4.58 actual vs 4.13
from actuals, 4.70 from chances).

## Best blend (weight on the chance-based rate)

| | 2024-25 best | 2025-26 best | Used |
|---|---|---|---|
| Assists | 0.5 | 0.5 | **0.5** (stable) |
| Rebounds | 0.7 | 0.2 | 0.3 (unstable; provisional) |

Re-checked after the gap-fill (79k player-games; 2024-25 and 2025-26 complete): same result:
assists best at 0.5 in both seasons (RMSE 1.842 / 1.820 vs 1.861 / 1.841 from actuals alone);
rebounds 0.7 vs 0.2. Very unlucky passers: next game 2.23 vs 1.81 from actuals, 2.43 from chances.

Rebound conversion is more skill/role (positioning, deferring) than assist conversion, so
rebound chances alone predict *worse* than actual rebounds.

**Angle:** a player whose recent assists lag his potential assists is likely priced off the
low actuals. The fair rate is ~halfway back. Worth adding to the angles list as "luck reversion".

## Not wired in yet
- 2026-27 single-day tracking queries return empty (the current season isn't split by date yet);
  season-to-date totals work. Plan: snapshot season totals daily and difference them to get
  per-game numbers going forward.
- Then apply the luck adjustment as a rate multiplier in the simulator / sheet (APM ×
  (0.5·expected + 0.5·actual) / actual over the last 10).
