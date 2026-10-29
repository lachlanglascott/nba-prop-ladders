# Context model: any teammate out (2026-10-25)

Code: `src/nba_ladders/absences.py`, `src/nba_ladders/context_sim.py`,
backtest `src/nba_ladders/backtest_context.py`, report `src/nba_ladders/ppm.py`.

```bash
uv run python -m nba_ladders.ppm "Tari Eason"     # full-strength minutes, PPM/RPM/APM, teammate unlocks
uv run python -m nba_ladders.backtest_context --seasons 22025 --out bt.parquet
```

## How it works

1. **Pair effects.** For each (player, teammate T): change in the player's minutes and in his
   per-minute rates (PPM, RPM, APM) in games T missed while rostered, pooled over shared seasons.
   T must normally play ≥12 min (fringe absences were noise).
2. **League prior** by position group (G / W / B), per minute T normally plays. Per 30 min of T:
   big↔big +3.3 min; other pairings +0.9 to +1.7. Small effects on rates (bigs' APM +6% when
   another big sits). Pairs are shrunk toward the prior (8 pseudo-games for minutes, 20 for rates);
   pairs with no shared history (rookies, new teammates) use the prior.
3. **Combining absences:** each pair's minutes effect floored at 0, scaled by
   `EFFECT_SCALE = 0.4`, summed with diminishing returns (18 × tanh(sum / 18)); rate
   multipliers multiply (clipped 0.7–1.4).
4. **Simulator:** each past game is un-adjusted for who sat that night, then re-adjusted for the
   target game's absences; minutes and rates still come from the same template game.

## Backtest (all players averaging 12+ min, effects refit monthly from prior games)

**Tuning season 2024-25**, choosing the effect scale (22,439 player-games):

| Scale | Log loss (context / current model) | Minutes MAE |
|---|---|---|
| 1.0 | 0.3022 / 0.3003 | 5.78 / 5.89 |
| 0.8 | 0.3003 / 0.3003 | 5.75 / 5.89 |
| 0.6 | 0.2992 / 0.3003 | 5.74 / 5.89 |
| **0.4** | **0.2986 / 0.3003** | **5.75 / 5.89** |
| 0.3 | 0.2986 / 0.3003 | 5.77 / 5.89 |
| 0.2 | 0.2989 / 0.3003 | 5.80 / 5.89 |

Raw with/without effects overshoot by ~2.5× (selection: players who gained minutes when a
teammate sat were often also trending up). 0.4 chosen (ties 0.3 on log loss, better MAE and
better in big-absence games).

**Holdout 2025-26** (α = 0.4, looked at once):

| Minutes freed by absences | Player-games | Log loss: context / current | MAE: context / current |
|---|---|---|---|
| < 10 | 2,065 | 0.3140 / 0.3106 | 6.42 / 6.56 |
| 10–30 | 3,869 | 0.2876 / 0.2873 | 5.72 / 5.81 |
| 30+ | 16,860 | **0.3004 / 0.3014** | **5.83 / 5.94** |
| All | 22,794 | **0.2995 / 0.2998** | **5.86 / 5.97** |

**Verdict: a modest, consistent improvement** in minutes accuracy, and in pricing when real
minutes are freed. It does **not** replace the starter-out backup-big model (still well
calibrated, see `simulator.md`); it extends coverage to every player and any combination of
absences.

## Recency weighting (added after the first holdout)

Templates are weighted by age: halves every `HALF_LIFE = 5` games, with `RECENCY_FLOOR = 0.2`
of weight spread evenly over all 120 games so rare big games still get sampled
(`sim.recency_weights`). Tuned on 2024-25 (half-life 60/30/15/8/5/3; floor 0–0.6).

| 2025-26 holdout | Log loss | Minutes MAE |
|---|---|---|
| Context model, no recency | 0.2995 | 5.86 |
| **Context model, recency + points fix** | **0.2835** | **4.82** |
| No-context model, recency + points fix | 0.2848 | 4.95 |

Recency is the biggest single gain in the project. Role (minutes) changes fast.

## Points bug fixed: odd totals

Points were drawn in 2-point units, so every simulated total was even and P(15+) = P(16+):
odd thresholds (and so every x.5 line from 14.5, 16.5…) were under-priced. Points are now drawn
as free throws + twos + threes (18/50/32% of points; variance ≈ 2.1× mean). 2025-26, 15+ pts:
6,493 → 6,918 predicted vs 7,426 actual.

## Role shift (current minutes level)

Recency weights keep a 20% floor over old games (protects tails) but lag role changes (a
33-min player now playing 25). `ROLE_WEIGHT = 1.0`: every template's minutes are shifted by
(minutes level over the last few games, half-life 3) − (weighted history mean), keeping each
game's minutes/production pairing.

| | 2024-25 (tuning) log loss / min MAE | 2025-26 (holdout) |
|---|---|---|
| Without | 0.28347 / 4.856 | 0.27959 / 4.818 |
| λ = 0.5 | 0.28254 / 4.755 | n/a |
| **λ = 1.0** | **0.28250 / 4.723** | **0.27822 / 4.676** |

## Global recalibration: tried, not adopted

`calibration.py` fits logit(p') = a + b·logit(p) (fit 2024-25: a = 0.018, b = 0.948). On 2025-26
it left log loss unchanged (0.27822 → 0.27822) and over-corrected the level (raw 1.5% low → 1.6%
high); long shots improved slightly (0.10202 → 0.10186), mid-range got worse. The role shift and
recency already removed most of the bias, so no global correction is applied.

## Remaining calibration gaps
- Point thresholds still ~4–8% low overall (2025-26: 10p 11,680 vs 12,150; 20p 3,759 vs 4,110),
  and recency trims backup bigs' normal-game tail (~14% fewer hits than actual vs 3% without).
  Next: a global calibration correction (e.g. isotonic on log-odds, fit on 2024-25).
- Backup bigs' starter-out tail swings by season (2024-25 under-predicted ~20%, 2025-26 over ~13%).

## Earlier open issue: stale history (now addressed by recency weighting)

Both models under-predict 15+ points (2025-26, 30+ freed: 5,528 actual vs ~4,700–4,760
predicted). A 120-game equal-weighted history underprices improving players. **Next: weight
recent games more** (e.g. exponential decay by game age), which likely matters more than
absence effects.

## Caveats
- Who was absent comes from box scores (rostered but didn't play) as a stand-in for the
  injury report; teammates' usual minutes are season averages (mild look-ahead).
- Effects are pooled across seasons; role changes within a season still leak in.
- Rate effects are small and noisy; minutes carry most of the signal.
