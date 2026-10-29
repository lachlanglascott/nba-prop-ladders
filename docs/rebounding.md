# Rebounding vs opponent, by position, vs expected (2026-10-30)

`uv run python -m nba_ladders.rebounding`. Code: `src/nba_ladders/rebounding.py`.

**Expected rebounds** per player-game = his offensive/defensive rebound rate from earlier games
(shrunk toward his position group's league rate, 150 pseudo-minutes) × actual minutes.
**Opponent factor** = actual / expected rebounds by players facing that opponent (shrunk toward
1 with 150 pseudo-rebounds). 155k player-games, 2019-20 → 2025-26.

## Is there a persistent opponent effect? (first half of season vs second half)

| | Groups | Corr H1→H2 | SD observed vs noise |
|---|---|---|---|
| DREB, all positions | 210 | **0.56** | 0.047 vs 0.030 → real, ~±4–5% |
| OREB, all positions | 210 | 0.48 | 0.057 vs 0.054 → borderline |
| DREB by position (G/W/B) | 630 | 0.40 | 0.054 vs 0.052 |
| DREB position residual after team effect | 630 | 0.25 | 0.040 |
| OREB position residual after team effect | 630 | 0.14 | 0.042 |
| Opponent misses/game (H1) → DREB factor (H2) | 210 | 0.31 | |

**Takeaways**
- The opponent effect is real and mostly **team-level**: miss volume (pace × shooting) and how
  hard they crash the offensive glass. 2025-26 example: HOU allowed 0.85–0.87× expected DREB to
  every position; WAS, IND, MEM conceded +4–11%.
- Position-specific effects exist beyond that but are weak (residual corr 0.25 DREB): use a
  heavily shrunk position adjustment on top of the team factor.
- "Vs expected" from tracking (chances × conversion) is in `docs/luck.md`.

## Worked example: Toumani Camara (POR vs MEM, 2026-10-29)
- 2026-27 so far: 23 rebound chances in 74 min (11.2/36, normal), converted 5 (22% vs his 50–62%
  in 2023-26; league 53%). 10 of 23 chances deferred to teammates (43% vs 8–19% before):
  Clingan and Robert Williams III (77% conversion) absorbing them.
- RW3 **Doubtful** (99% sit rate) removes the deferral sink and frees ~18 big minutes.
- MEM allowed wings +6% DREB vs expected in 2025-26.
- Rough fair rebounds ≈ 0.31 chances/min × ~27 min × ~50% ≈ 4.2 before the RW3 and MEM bumps, vs a
  line likely set off his 1.7/game start. Luck reversion (angle G) + rebounder out + soft opponent.

## In the model (enabled)

`opponent_multipliers()`: as-at team factor (this season + last season at 0.5 weight, shrunk
with 150 pseudo-rebounds) × position residual (shrunk with 400), rescaled to average 1 per
position group, blended by the player's own OREB/DREB mix. Applied as a rebound-rate
multiplier in the sheet (alerts and minutes-unlocked) and `price --opponent XXX`.

Backtest (context model, all players 12+ min, refit monthly), log loss on rebound lines:

| Line | 2024-25 (tuning) without → with | 2025-26 (holdout) without → with |
|---|---|---|
| 5r | 0.51259 → 0.51244 | 0.51083 → **0.50939** |
| 8r | 0.31305 → 0.31230 | 0.30954 → **0.30854** |
| 10r | 0.20026 → 0.20003 | 0.18934 → **0.18854** |
| 12r | 0.11936 → 0.11914 | 0.10734 → 0.10734 |
| 10p/10r | 0.17729 → 0.17683 | 0.16675 → **0.16604** |
| 15p/8r | 0.21421 → 0.21359 | 0.20759 → **0.20657** |
| 15p/8r/2a | 0.18638 → 0.18598 | 0.18111 → **0.18014** |
| 20p/5r/5a | 0.13983 → 0.13983 | 0.12847 → 0.12861 |
| **All rebound lines** | 0.23287 → **0.23252** | 0.22512 → **0.22440** |

Small but consistent; the holdout gain is twice the tuning gain. A first version without the
rescale-to-1 step biased all predictions down (expectation model runs slightly high).
