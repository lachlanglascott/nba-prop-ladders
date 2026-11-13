# Usage redistribution when teammates sit (2026-11-13)

Code: `src/nba_ladders/usage.py`. Run `uv run python -m nba_ladders.usage` for the fit and backtest, or
`--player "VJ Edgecombe" --out "Joel Embiid,LeBron James" --date YYYY-MM-DD` for one player tonight.

**Why.** Spots already moves minutes when a teammate sits, and takes per-minute rates from games that match tonight's
lineup. With few matching games, though, the rate bump was missed. Last season's games only counted if tonight's
absentees were also missing back then. LeBron wasn't on PHI in 2025-26, so Edgecombe's 38 games without Embiid didn't count.

**Raw pattern** (2021-27, 15+ min, vs the player's own season rate). Freed usage per game from absent rotation players:
15-30 gives usage/min 1.00 and pts/min 1.01; 30-45 gives 1.04 / 1.04; 45+ gives 1.09 / 1.08. All of this is on top of
+1 to +5 extra minutes.

## Model

Multiplier on player i's per-minute stat = 1 + the sum, over tonight's absent teammates T, of effect(i, T):

- **Prior**, fitted on 2019-24 only: beta × T's usual volume / 100 × (i's own rate / league rate).
  - Points: beta 0.20 (same position) / 0.17 (different position), per 100 of T's usage (FGA + 0.44 FTA + TOV).
  - Assists: 0.65 / 0.50, per 100 of T's assists.
  - Rebounds: **rebound share**, 0.59 (same) / 0.78 (diff) per 100 of T's rebounds *above 5 a game*, with no tilt.
    Only a big rebounder's absence frees boards: per-minute rebounds +2-6% at 8+ rpg, ~0 below (2023-27 tracking check).
    Rebound chances barely move, so the gain is in conversion (fewer contested boards). Embiid out ≈ +3-4% for each teammate.
- **Player-specific**: i's residuals in past games T missed, across any season and any lineup, shrunk toward the prior
  with K pseudo-games (pts and ast 80, reb 160).
  - Residuals are measured against i's absence-neutral trailing rate: each history game is divided by its own prior multiplier.
  - Single games are noisy (pts ±45%) next to real pair effects (±5%), hence the heavy shrinkage.
- **Absent** means T played 3+ of the team's last 10 games this season at 20+ min and missed this one. The window stays
  inside the season, otherwise departed players count as absent in October.

## Backtest

Poisson deviance vs his plain trailing 40-game rate, with actual minutes, using earlier games only:

| season | pts: prior | pts: player-specific | ast: prior | ast: player-specific | reb: prior (rebound share) |
|---|---|---|---|---|---|
| 2024-25 (K tuned here) | −408 | **−493** | −61 | **−116** | −24 |
| 2025-26 (held out) | −433 | **−534** | −68 | **−115** | −27 |
| 2026-27 so far | −102 | −97 | −13 | −12 | −6 |

## In spots

Tonight's new absences (not the established ones; those are already in his current-role games) scale the rate prior
that the matching games are blended with: pts36, potential assists and rebound chances. With 3+ matching games his
own sample dominates; with 0-2 the usage model carries the bump. The projection line shows it, e.g. "usage without them: pts +6%, ast +3%".

Examples on 12 Nov:
- **Edgecombe, Embiid + LeBron out:** pts +6%, ast +3%. Without Embiid he has 28 past games: own +4.0% vs prior +3.8%.
  - Caveat: the raw split says +20% pts/min, but Embiid missed half of 2025-26, so his rising rookie baseline is
    already mostly Embiid-out games.
- **Poeltl, Barnes out:** pts +2%, ast −2% (20 games without Barnes).

## Touches spot (spots `touches`)

A new absence where his minutes stay flat (under +2) but the usage model says his per-minute pts, ast or reb go up.
- It needs **+2% backed by 3+ of his own games without them**, or **+4% on the prior alone**. On the league prior
  alone, +2-3% fires on almost any absence.
- Sportsbet 2024-26: rungs with a +2-7% bump returned −5 to −7%, against −12 to −15% with no bump, and −4% with a good
  angle H spot. Star-sized bumps (+7%+) are priced as news (−11%). Not a standalone edge, so it's graded **C**, or
  **B** when angle H boosts the bet.
