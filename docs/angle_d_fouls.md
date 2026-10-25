# Angle D: foul-trouble opponent — findings (2026-10-25)

Data: 2025-26 regular-season play-by-play, 1,204 / 1,230 games (pull stopped itself after 3
consecutive failures; remaining 26 can be re-tried with
`uv run python -m nba_ladders.sources.pbp --season-id 22025`, which skips cached games).
Code: `src/nba_ladders/sources/pbp.py` (fetch/parse), `src/nba_ladders/fouls.py` (analysis,
`uv run python -m nba_ladders.fouls`). Built by a background agent; reviewed in the main session.

## Method

- **Starters from play-by-play** (players active in Q1 before being subbed in): exactly 5 per
  team in 99% of team-games. Starting centre = most centre-like starter (C > C-F > F-C).
- **Early foul trouble** = ≥2 PF in Q1 or ≥3 PF before half (personal fouls only).
- Check: play-by-play PF totals match box-score PF in **99.7%** of 26,117 player-games.
- **Note:** foul trouble happens *during* the game, so it isn't a pre-game trigger. The betting
  question is whether it is **predictable** (from the centre or the opponent), and whether it
  moves enough minutes to matter.

## Result 1: when it happens, the backup barely benefits

| | Team-games | Starter min | Primary backup min | Backup 10/10 |
|---|---|---|---|---|
| No early trouble | 1,825 | 28.5 | 16.8 | 4.8% |
| Early trouble | 301 | 25.9 | 17.9 | 4.9% |

By starter's first-half PF: 0 → 16.7 backup min, 2 → 17.0, 3 → 17.6.
Within-player (same backup, trouble vs clean games): **+0.7 min**.

Compare starter **out**: +5 min and 10/10 doubles (`docs/angles_abc_findings.md`). Coaches
manage foul trouble (sit him in Q2, ride him after half), and the lost minutes are spread
across the roster rather than going to the backup big.

## Result 2: foul-proneness is partly predictable

| Split-half check (season halves) | Corr H1 vs H2 | Spread vs pure noise |
|---|---|---|
| Per starting centre (23 centres, ≥15 games each half) | 0.39 | 0.065 vs 0.046 |
| Per opponent (30 teams) | 0.19 | 0.053 vs 0.041 |
| Opponent FTA/game (H1) → induced trouble rate (H2) | 0.38 | |

Most foul-prone starting centres: Duren (DET) 28%, Poeltl (TOR) 26%, Gafford (DAL) 26%,
W. Carter Jr. (ORL) 25%; league 14%.
Opponents most likely to put the centre in early trouble: SAS, MIN, ORL, HOU (~21–23%);
least: BOS, CLE, IND (~6%). Opponent FTA/game predicts this better than the opponent's own
past rate.

## Verdict

**Weak for the backup-big ladder thesis.** Foul trouble is somewhat predictable, but it moves
only ~1 minute to the backup, so it won't create 10/10 or tail outcomes. Better uses:
- **Small simulator input:** shift the minutes mix slightly for backups of foul-prone centres
  against high-FTA opponents. Expect a small effect.
- **Possibly the other side:** starter *unders* (minutes, rebounds) against high-FTA opponents.
  Not tested.

## Caveats
- One season only. Opponent rates rest on ~70 games each, so noisy.
- Backups who didn't play (DNP) aren't in the box table, so "backup minutes" excludes zeros.
- Late-game foul-outs (5–6 PF in Q4) aren't covered; that's a different, rarer scenario.
