# Model vs market backtest: 2025-26 Sportsbet (2026-10-31)

Data: The Odds API historical, ~2h before tip, 491 games / 65 game days of 2025-26 (random day
sample), 183,132 Sportsbet "X+" rungs (points, rebounds, assists, P+R totals) with actual
results. Model = plain simulator as-at (role shift, recency, luck rebuilt from per-game
tracking, opponent rebounds). Not the absence-context model. Tune/test split by random game day.
Code: `market_backtest.py` (build/report), `market_blend_check.py`.

## 1. Accuracy: Sportsbet is better than the model on its own

| | Log loss |
|---|---|
| Sportsbet implied (with margin) | 0.5630 |
| Model alone | 0.5826 |
| Book + model (logistic) | 0.5629 |

Model coefficient given the book: 0.063 (SE 0.010). Real but small information overall.
By stat: rebounds 0.193 (SE 0.026), points 0.068, assists 0.058, P+R ≈ 0.
**Within angles it's much stronger:** starter-out 0.242 (SE 0.053), luck 0.392 (SE 0.074).

## 2. Betting the model's value flags at Sportsbet: loses

| Strategy (test half) | Bets | ROI | 90% CI |
|---|---|---|---|
| Raw model flags, tuned cushions | 8,333 | **−7.4%** | −12.7% to −1.9% |
| Raw model flags, default cushions | 7,343 | −8.1% | −14.7% to −1.3% |
| Market-anchored blend, 2% cushion | 701 | −0.3% | −23% to +25% |
| Blend at best AU price (line shopping), 2% | 3,405 | −4.4% | −13% to +5% |

The model's raw flags are adverse-selected: it is most wrong exactly where it disagrees with
the book. The blend is more accurate than the book (held-out log loss 0.5601 vs 0.5673), but
the information is smaller than the margin on single legs. Best AU price averages only +2.9%
over Sportsbet (PointsBet AU most often best).

## 3. Verdict

**Single-leg AU player props are efficiently priced relative to this model. No single-leg
strategy shows profit on held-out games.** The model knows things the book doesn't
(starter-out, luck reversion, rebounds) but not enough to beat the margin one leg at a time.

## 4. Ladder tails and variance (blind, no model; `ladder_tails_check.py`)

Betting *every* Sportsbet rung, by price band. Strong longshot bias: the margin grows with odds.

| Band | Rungs | Blind ROI | Implied → actual |
|---|---|---|---|
| <$2 | 93,448 | −8% | 74% → 68% |
| $2–3 | 35,079 | −12% | 42% → 37% |
| $3–5 | 32,683 | −15% | 27% → 23% |
| $5–10 | 18,249 | −21% | 16% → 13% |
| $10–20 | 2,846 | −35% | 8.5% → 5.5% |

Split by as-at minutes variance (sd of minutes over the previous 20 games; terciles 4.5 / 5.8):
high-variance players' tails are **less** overpriced ($5–10: −18% vs −26% for low variance),
the direction the thesis predicts, but still clearly negative. The model adds nothing in
high-variance tails (coef 0.04 ± 0.05).

**Thesis spots, blind tails (price ≥ $3):**

| Spot | Bets | Hit | Avg price | Blind ROI | 90% CI |
|---|---|---|---|---|---|
| **Starter out × high variance** | 463 | 24.6% | $4.98 | **+9.5%** | −18% to +36% |
| Starter out, reb & P+R tails | 461 | 23.9% | $4.99 | +4.9% | −21% to +35% |
| Luck spots | 588 | ~19% | ~$4.9 | −10% to −18% | |
| Everything else | ~30,000 | ~18% | ~$5.1 | −18% to −22% | −28% to −13% |

**The one segment consistent with the thesis:** tail ladders on high-variance players when the
starting centre is out beat the general pool by ~28 points of ROI. The CI still includes zero
(463 bets, one season sample), so this is a lead to confirm with more starter-out games.

## Where the edge may still be (not testable with single legs)

1. **SGM correlation pricing:** the original thesis. Books are efficient on each leg; the
   question is whether their SGM correlation adjustment is too small for backup bigs / P+R.
   Test: market-anchored joint = devigged leg prices × model lift, vs quoted SGM prices
   (screenshots). ~50–100 quotes on high-lift spots.
2. **Timing:** snapshots here are ~2h before tip, after most injury news. The "book is slow"
   edge needs prices right after a NEW absence vs later. Cheap live test: snapshot when the
   sheet flags a new starting-centre absence, then again ~30–60 min later.
3. **The context model** (absences) wasn't used. Starter-out spots had only 2,594 rungs.

## Caveats
One season sample (40% of games), AU history limited to some books, Betr has no history.
