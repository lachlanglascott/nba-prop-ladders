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
