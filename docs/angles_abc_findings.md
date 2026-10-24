# Angles A–C: findings (2026-10-24)

Data: regular seasons 2019-20 → 2025-26. Code: `src/nba_ladders/scenarios.py`, `src/nba_ladders/angles_abc.py`.

## Method

- **Scenario = something knowable pre-game.** Earlier screens used "25+ minutes", which is an
  outcome (look-ahead). Here the trigger is **the team's starting centre is out** (on the roster
  but didn't play), which the injury report tells you before tip.
- **Starting centre** = the team-season's highest-minutes big (C / C-F / F-C) who rebounds like a
  centre (≥10 reb/36). Position labels alone fail: Plumlee and Markkanen are both "F-C".
- **Backup big** = any other big (20+ GP) averaging fewer minutes than the starter.
- **Predictability test:** does a player's *previous* season predict his *next* season?
  If not, the effect may be real on average but you can't pick players for it in advance.

Sample: 14,297 backup-big games, 2,581 with the starter out, 343 player-seasons.

## League-wide: the starter-out scenario

| Backup bigs | Games | Minutes | 10/10 | 12/12 | 15/12 | P(12/12 \| 10/10) | P(15/12 \| 10/10) |
|---|---|---|---|---|---|---|---|
| Starter plays | 11,716 | 15.5 | 5.7% | 2.1% | 1.4% | 0.37 | 0.25 |
| **Starter out** | 2,581 | **20.5** | **11.6%** | **5.9%** | **4.3%** | **0.51** | **0.37** |

The whole distribution shifts **and the tail gets fatter**: 10/10 doubles, but 15/12 triples.
Scaling a player's average up by the extra minutes would under-price the top rungs in exactly
these games. This is the original thesis, confirmed on 7 seasons of data.

## A — Conversion: **predictable**

Previous-season 10/10 rate (normal games) → next-season 10/10 rate *in starter-out games*
(130 player pairs, corr 0.35):

| Previous-season tercile | Players | Starter-out games | Next-season 10/10 when starter out |
|---|---|---|---|
| low (0%) | 69 | 640 | 9.2% |
| mid (~2%) | 18 | 186 | 9.1% |
| **high (~12%)** | 43 | 459 | **17.0%** |

Players who converted last season convert again when the door opens: nearly double the rate.
This is the selection rule for angle A.

## B — Tail: **real as a scenario effect; unproven as a player trait**

- Scenario-level: tail ratio rises from 0.37 → 0.51 when the starter sits (table above).
- Player-level: only **16** player pairs had ≥5 10/10 games in back-to-back seasons. Correlation
  0.20, too few pairs to conclude. "Ware/Okongwu are tail players" is a hypothesis, not a finding.
- Next: model the tail structurally (minutes ceiling × per-minute rates × pts/reb correlation,
  simulated) rather than as a per-player ratio that needs years of 10/10 games to estimate.

## C — Injury trigger (minutes uplift): **real on average, not a stable player trait**

Previous-season uplift → next-season uplift (102 pairs, corr 0.14):

| Previous-season uplift tercile | Avg previous uplift | Next-season uplift |
|---|---|---|
| low | 2.3 min | 6.3 min |
| mid | 7.4 min | 8.1 min |
| high | 12.7 min | 7.0 min |

The jump is ~6–8 minutes regardless of last season. It's driven by the **current roster
situation** (who else can soak up the minutes), not by the player. So angle C has to be
modelled per game from the depth chart, not from the player's history.

## Your players: starter in vs out

| Player | Season | Starter | In: games / min / 10/10 | Out: games / min / 10/10 |
|---|---|---|---|---|
| Drummond | PHI 21-22 | Embiid | 37 / 14.8 / 5% | 12 / 29.3 / 42% |
| Drummond | CHI 23-24 | Vučević | 74 / 16.2 / 19% | 5 / 29.4 / 80% |
| Drummond | PHI 25-26 | Embiid | 22 / 11.3 / 9% | 38 / 23.8 / 34% |
| Okongwu | ATL 22-23 | Capela | 63 / 20.6 / 11% | 16 / 33.1 / 31% |
| Okongwu | ATL 23-24 | Capela | 48 / 24.3 / 10% | 7 / 33.9 / 43% |
| Ware | MIA 25-26 | Adebayo | 68 / 21.1 / 28% | 9 / 30.1 / 56% |
| Bitadze | ORL 23-24 | Carter | 36 / 9.3 / 3% | 26 / 24.0 / 15% |
| Bitadze | ORL 24-25 | Carter | 57 / 19.5 / 19% | 13 / 25.0 / 31% |
| Richards | CHA 22-23 | Plumlee | 41 / 17.0 / 20% | — (Plumlee traded, never "out") |

## New angle E — role change (trade / promotion)

Richards' best run came after Plumlee was **traded** (Feb 2023), which is a permanent role change,
not a one-game absence, so it isn't in the starter-out numbers. Same family as the Robinson
"new starter" bet. Worth its own test: how quickly does the market reprice after a trade or
depth-chart promotion?
