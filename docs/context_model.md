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

## Biggest open issue found: stale history

Both models under-predict 15+ points (2025-26, 30+ freed: 5,528 actual vs ~4,700–4,760
predicted). A 120-game equal-weighted history underprices improving players. **Next: weight
recent games more** (e.g. exponential decay by game age), which likely matters more than
absence effects.

## Caveats
- Who was absent comes from box scores (rostered but didn't play) as a stand-in for the
  injury report; teammates' usual minutes are season averages (mild look-ahead).
- Effects are pooled across seasons; role changes within a season still leak in.
- Rate effects are small and noisy; minutes carry most of the signal.
