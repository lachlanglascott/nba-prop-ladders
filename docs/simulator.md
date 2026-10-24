# Simulator (Phase 2)

Code: `src/nba_ladders/sim.py`, backtest `src/nba_ladders/backtest.py`, CLI `src/nba_ladders/price.py`.

```bash
uv run python -m nba_ladders.scenarios                       # rebuild scenario table
uv run python -m nba_ladders.backtest                        # as-at backtest 2024-25 + 2025-26
uv run python -m nba_ladders.price "Andre Drummond" --starter-out
```

## How it works

Each simulated game resamples one of the player's last 120 backup-role games as a *template*:

1. **Minutes** = the template's minutes (± 2 min jitter). In a starter-out game, templates come
   from his own starter-out games, topped up (8 pseudo-games) with starter-in games scaled by the
   league uplift ×1.32. The uplift is situational, not a player trait (angle C), so borrowing the
   league figure is justified.
2. **Rates** = the *same* template's pts/min and reb/min. So a game where he was playing well
   brings both its minutes and its production, and pts/reb keep their within-game link. Rates are
   shrunk toward his mean to remove sampling noise (added back in step 3).
3. **Counts** ~ Poisson(rate × minutes), points in 2-point units.

Output: P(pts ≥ a and reb ≥ b) per rung, fair odds, and the same if the legs were independent.

## Backtest: 2024-25 + 2025-26, as-at (only prior games used), 4,621 backup-big games

Starter-out games (845), hits per rung:

| Rung | Actual | Simulator | Independent legs |
|---|---|---|---|
| 8/8 | 218 | 212 | 178 |
| 10/10 | 119 | 118 | 93 |
| 12/12 | 65 | 61 | 45 |
| 15/12 | 40 | 39 | 26 |
| 15/15 | 14 | 18 | 11 |
| 20/15 | 7 | 10 | 6 |

All games: 1,859 actual hits vs 1,846 simulated vs ~1,430 independent. Log loss beats the
independent baseline in both scenarios (starter out 0.230 vs 0.237; plays 0.135 vs 0.143).

**Independent legs under-price the ladder by ~30–50%** in starter-out games (e.g. 15/12 fair
$32 vs $21 actual). That's the edge the thesis is about, if books price closer to independence.

Calibration by predicted probability:

| Predicted | n | Actual |
|---|---|---|
| ≤2% (avg 0.5%) | 22,571 | 0.9% |
| 2–5% | 5,248 | 3.8% |
| 5–10% | 3,810 | 7.1% |
| 10–20% | 2,831 | 14.3% |
| 20–35% | 1,476 | 21.7% |
| 35–50% | 744 | 37.1% |
| >50% | 312 | 55.8% |

Well calibrated in the 5–20% band where most ladder bets sit. Slightly **under**-confident on
long shots and **over**-confident above 20%. Fair for long shots is a bit too long, and for short
prices a bit too short.

## Ablations (how we got here)

| Version | Starter-out hits predicted (actual 503 in that run) |
|---|---|
| minutes and rate drawn independently, +5 min uplift | 397 |
| paired minutes+rate, +5 min | 411 |
| **paired, ×1.32 uplift (current)** | 456 → 616/621 after the centre fix |

Caveat: these choices were made looking at the same test seasons, so expect slightly worse
results out of sample. 2026-27 is the clean holdout; re-run as the season goes.

## Known limitations / next
- **History = backup-role games only.** Missi's 2024-25 (when he started) is excluded, so his upside
  is under-represented. Next: use all games with a role flag.
- Backup role uses full-season averages, which is mild hindsight in the backtest.
- No opponent, pace, home/away, rest, or foul-trouble inputs yet (angle D will feed in).
- Rungs fixed at P+R; add PRA, rebounds-only ladders.
