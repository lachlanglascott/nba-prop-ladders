# Phase 1 findings (2026-10-24)

Data: regular-season player game logs 2023-24 → 2026-27 (3 × 1,230 games + 28 games of 2026-27).

## The two week-1 examples

| Player | 2026-27 team | Games | Line |
|---|---|---|---|
| Mitchell Robinson | BOS (from NYK) | 10-20 @DET; 10-23 vs NYK | 18 min 4p/3r; 22 min 4p/4r (5 PF) |
| Andre Drummond | NYK (from PHI) | 10-20 vs PHI; 10-23 @BOS | 24 min 12p/14r; 7 min 2p/2r |

## Ladder hit rates, 2023-24 → 2025-26, split by minutes

| | Minutes | Games | 10p & 10r | 15p | corr(pts, reb) |
|---|---|---|---|---|---|
| Robinson | <25 | 81 | 7% | 1% | 0.42 |
| Robinson | ≥25 | 27 | 15% | 7% | 0.38 |
| Drummond | <25 | 143 | 11% | 3% | 0.56 |
| Drummond | ≥25 | 39 | **62%** | 21% | 0.51 |

**Takeaway:** the thesis needs *two* conditions, not one.
1. **Minutes scenario:** the player gets 25+ minutes. Robinson's "new starter" role gave only 18–22 so far.
2. **Production scales with minutes:** Drummond's does (10/10 hit rate goes 11% → 62%); Robinson's barely moves (7% → 15%), and he almost never scores 15. Robinson is a low-usage big, so extra minutes become rebounds, not points. A 15-point rung was very unlikely even in his best scenario.

Drummond's pts/reb correlation (~0.5) is also higher than Robinson's, so his P+R multis benefit more from correlation.

## Teammate-absence (with/without)

- Drummond: Vučević out (CHI 23-24) 16 → 29 min, 14p/19r; Embiid out (PHI 25-26) 11 → 24 min.
- Robinson: Towns out (NYK 25-26) 19 → 25 min, 10p/12r.

## Caveats / next steps
- With/without is confounded by role changes over a season (the teammate's absence may coincide with tanking, trades or a rotation change). Next: compare against the player's minutes in *nearby* games, not season averages.
- Minutes ≥25 is a crude scenario; next step is a proper minutes mixture model.
- Need play-by-play for foul trouble (e.g. Robinson's 5 PF on 10-23).
