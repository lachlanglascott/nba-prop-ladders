# Opposition by position: expected vs actual, predictive, priced? (2026-11-13)

Code: `src/nba_ladders/opp_position.py` (`uv run python -m nba_ladders.opp_position [--player X]`;
`table(as_of)` = tonight's factors). Prompted by Poeltl: 19/11 vs MIL, 18/7 vs MEM, then 6/9 vs BOS.

**Method.** Each player-game (10+ min, 10+ prior games): expected = his own per-minute rate over his
previous 40 games x actual minutes, so this measures rate, not minutes. Opponent factor = actual /
expected allowed, using only earlier games, at three levels of "similar player": team, G/W/B, and
archetype (lead / scoring guard, wing / forward, stretch / rim big). Each level is shrunk toward the
one above it, and last season's factor (regressed halfway) is the early-season prior.

## 1. The opponent effect is real, and G/W/B is the right level of detail

Slope of actual/expected on the as-at factor, 2020-21 to 2026-27 (~112k player-games; 1.0 = the factor is fully real):

| | pts | reb | ast |
|---|---|---|---|
| team | 1.48 | 1.26 | 1.30 |
| G/W/B | **1.00** | **0.85** | **0.88** |
| archetype | 0.77 | 0.65 | 0.69 |

What each finer level adds beyond the coarser one (slope ± SE):

| | pts | reb | ast |
|---|---|---|---|
| G/W/B over team | 0.71 ± 0.07 | 0.48 ± 0.07 | 0.56 ± 0.07 |
| archetype over G/W/B | 0.15 ± 0.07 | 0.03 ± 0.07 | 0.16 ± 0.07 |

So "guards vs MIL" is real, but "lead guards vs MIL" as distinct from "guards vs MIL" is mostly noise.
Splitting guards or bigs further leaves samples too small to say anything. Typical size: the G/W/B
factor has an SD of ~4-5%, so a strong spot is ±6-10%. That is real, but small next to one game's
noise: Poeltl's 19 vs MIL and 6 vs BOS are mostly variance around a ~+4% and a ~-9% spot.
`matchup.group_factors` (used by spots) correlates 0.87-0.93 with this.

## 2. Sportsbet barely prices it

Logistic regression on 657k Sportsbet X+ rungs (2024-25 + 2025-26): hit ~ logit(book) + c·log(factor),
SE clustered by game. If the book fully priced the opponent, c would be 0.

| factor | c | SE |
|---|---|---|
| team | 0.99 | 0.33 |
| G/W/B | **0.82** | 0.24 |
| archetype | 0.66 | 0.20 |

Holds by season (G/W/B: 0.68 in 24-25, 0.90 in 25-26) and on top of our model (0.94 ± 0.24 with logit(p_model) in).
By stat: **ast 1.11 ± 0.27**, reb 0.86, P+R 0.85, pts 0.53 ± 0.32. So the book ignores most of the opposition-by-position effect,
assists most of all. **On its own it is below the margin:** overs at Sportsbet with factor >= 1.06 lose 5-16%, against 6-19% for
everything else. Even at the best AU price (~+3%), a single leg on this alone still loses.

## 3. Where it matters: stacked on an existing angle

ROI at Sportsbet, rungs clustered by player-game for the bootstrap 90% CI:

**Starter-out backup bigs (the headline angle), split by opponent factor for the bet's stat**

| factor | player-games | ROI | 90% CI | 24-25 | 25-26 |
|---|---|---|---|---|---|
| all | 386 | -5.7% | -12% to +1% | +0.6% | -9.0% |
| **< 1.00 (bad spot)** | 262 | **-14.3%** | **-22% to -6%** | | |
| >= 1.02 | 235 | +8.5% | -1% to +19% | +19% | +3% |
| >= 1.04 | 173 | +9.0% | -2% to +20% | +12% | +8% |

**Assist luck (luck >= 1.10) x opponent ast factor**

| factor | player-games | ROI | 90% CI | 24-25 | 25-26 |
|---|---|---|---|---|---|
| all | 429 | +1.0% | -7% to +8% | | |
| < 1.00 | 209 | -2.5% | -13% to +8% | | |
| >= 1.04 | 83 | **+21.6%** | **+4% to +41%** | +10% | +37% |

Thresholds were chosen after looking at the data, but ROI rises steadily with the factor and both seasons agree.
These samples are still small, so treat this as an angle under test, not a proven edge.

## Angle H: opposition spot

- **Gate (strongest evidence):** a starter-out backup big in a bad opposition spot for the bet's stat (factor < 1.00) is a pass.
  This is the BOS case: BOS vs bigs pts 0.91.
- **Boost:** an angle spot (starter out / role / assist luck) with factor >= 1.04 for the stat gets more confidence, especially assists.
- **Never alone:** the opponent factor by itself is under the margin.
- Not yet tested: role spots (teammate-out promotions like Poeltl with Barnes out) aren't in the market backtest's `starter_out`.

## 4. In the model (2026-11-13)

The spots projection now uses these factors for **every stat and every position group**. Each per-minute rate is multiplied
by factor^w; matching games are de-biased the same way, and so are the anchored "hit in games like tonight" counts. The
weights w are the pooled 2020-27 slopes:

| | pts | reb | ast |
|---|---|---|---|
| G | 0.93 | 0.82 | 0.98 |
| W | 0.92 | 0.82 | 0.82 |
| B | 1.12 | 0.95 | 0.55 |

This replaces `matchup.group_factors` with pts x1.5 and guards' reb x1.3 (weights fit to that more-shrunk estimator).
Poisson deviance vs no opponent adjustment, by season: 2024-25 pts -343 / reb -88 / ast -224; 2025-26 -474 / -220 / -178;
2026-27 so far -79 / -19 / -18. Weight 1.0 does about the same, so the factor is well calibrated.
`matchup.group_factors` / `player_multipliers` / `weakness_line` now return these factors too, so `candidates` and `price` agree with spots (weights `opp_position.OPP_W`).

**Angle H stays as a separate overlay.** The model moves the projection, but honest fair moves only part of the way
(confidence 0.08-0.35) from the market toward it. The gate/boost rules are tested against the book directly, so they
apply on top: bigs' P/R pass below 1.02 and boost at or above; guards'/wings' assists down below 0.97 and boost at or
above 1.02 on an angle spot; wings' points down below 0.97; assist luck x opp ast >= 1.04 boost.
