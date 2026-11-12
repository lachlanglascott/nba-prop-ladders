# Angles

Each angle is a separate hypothesis with its own trigger, target market, and test. Don't blend
them: a player can be a good fit for one and a terrible fit for another.

| # | Angle | Thesis | Key metric | Market | Status |
|---|---|---|---|---|---|
| A | **Conversion** | When minutes come, production reliably follows | P(10/10 \| starter out) | P+R SGM at moderate lines | **predictable** from prior season ([findings](angles_abc_findings.md)) |
| B | **Tail / "goes huge"** | Low hit rate, but when it hits the upper rungs hit together; priced as if independent/normal | P(higher rung \| 10/10), fair $ per rung vs offered $ | P+R ladder, long odds | **scenario effect confirmed**; player trait unproven (16 pairs) |
| C | **Injury trigger** | Deep-bench big whose minutes jump only when the starter sits | minutes with blocker in vs out | anything, *if* in before the book reprices | real (~6–8 min) but **situational**, not a player trait |
| D | **Foul-trouble opponent** | Opponent bigs foul a lot, so the starter sits and the backup plays | opponent big PF/36, starter early-foul rate | backup's P+R | **weak**: partly predictable, but moves only ~1 min to the backup ([findings](angle_d_fouls.md)) |
| E | **Role change** | Trade or depth-chart promotion makes a backup the starter (Richards after the Plumlee trade, Robinson at BOS) | production in new role vs prior; how fast lines reprice | P+R in first games of new role | not started; Lendeborg (GSW rookie, 29–37 min from game 1) logged |
| F | **All-round P/R/A** | Big-minutes forward who fills every column (old Larry Nance bets, Lendeborg); three legs that all move with minutes | 3-leg joint vs independent | P/R/A SGM | **confirmed in backtest**: independent legs predict ~half the hits ([simulator](simulator.md#three-leg-pra-combos)) |

| G | **Luck reversion** | Recent assists (rebounds) lag potential assists (rebound chances); lines set off the low actuals | luck over last 10 vs next game | AST (REB) props, P/R/A | **confirmed for assists** (~halfway reversion, stable across seasons); rebounds weaker ([findings](luck.md)) |
| H | **Opposition spot** | Book ignores most of what an opponent allows to a position group (G/W/B), most of all bigs' P/R and assists | as-at opp x group factor for the bet's stat | **in the model** (all stats/groups) **and** an overlay: bigs' P/R pass < 1.02 / boost ≥ 1.02, G/W ast and wing pts down < 0.97 | bigs' P/R < 1.02 −12 to −14% vs ≥ 1.02 −0.6% (P+R), both seasons; luck × good ast spot +22% (83 games) ([findings](opp_position.md)) |

## Market check (2025-26 Sportsbet backtest, docs/market_backtest_findings.md)
Single legs are efficiently priced: no single-leg strategy profits on held-out games. The model
adds real information in starter-out (coef 0.24) and luck (0.39) spots and on rebounds, but less
than the margin. Remaining edge candidates: **SGM correlation pricing** and **timing after news**.

**Angle check vs Sportsbet, two seasons (664k rungs):** the 2025-26 backup-big leads were noise
(−11% to −21% on the full sample). Only **potentials luck on assists** holds: luck ≥ 1.10 at the
best AU price +3.7% (+3.8% / +3.5% by season, CI includes 0). Details:
docs/market_backtest_findings.md §6.

## A vs B are different players

- **Conversion (A)** fits players with a high 10/10 rate *at* big minutes: Ware, Bitadze, Richards, Drummond.
- **Tail (B)** fits players whose 10/10 is rare but, once hit, keep going.
  The ratio `P(12/12) / P(10/10)` is the "keeps going" measure.

## First look at the tail angle (2019-20 → 2025-26, seasons with ≥20 GP)

| Player-season | 10/10 rate | P(12/12 \| 10/10) | P(15/12 \| 10/10) | Fair $ 10/10 → 12/12 → 15/12 |
|---|---|---|---|---|
| Ware MIA 25-26 | 0.31 | 0.71 | 0.54 | 3.2 → 4.5 → 5.9 |
| Okongwu ATL 24-25 | 0.38 | 0.68 | 0.50 | 2.6 → 3.9 → 5.3 |
| Missi NOP 24-25 | 0.19 | 0.36 | 0.36 | 5.2 → 14.6 → 14.6 |
| Richards CHA 22-23 (with Plumlee) | 0.22 | 0.29 | 0.07 | 4.6 → 16.2 → 65 |
| Richards CHA 23-24 (Plumlee gone) | 0.28 | 0.16 | 0.05 | 3.5 → 22.3 → 67 |

Early reads, small samples (5–28 10/10 games per row):
- **Richards was a *conversion* player, not a tail player.** He hit 10/10 often but rarely went
  past 12/12, so his edge was the 10/10 rung, not the ladder.
- **Ware 25-26 and Okongwu 24-25 look like real tail players**: over half their 10/10 games
  went on to 15/12.
- **Missi** is between the two. At $10 for 10/10 the 24-25 rate (fair ~$5.2) would be strongly +EV,
  but his 25-26 rate fell to 0.08 (fair ~$13) when his minutes dropped. The scenario matters more than the player.

## Open questions per angle
- **B:** season rates mix scenarios. Re-run conditional on the bet trigger (blocker out, 25+ projected
  minutes). We need *offered* prices to confirm EV; no historical odds yet, so log quotes going forward.
- **C:** de-confound with/without from rotation changes (compare to nearby games).
- **D:** pull play-by-play for foul timing.

## Player examples from the user's betting (for testing the any-teammate-out model)

| Player | Type | Why | Data |
|---|---|---|---|
| Nick Richards (CHA) | backup C → starter | minutes with/after Plumlee | conversion (A), role change (E) |
| Andre Drummond | backup C | Embiid/Vučević out → 24–29 min, 10/10 34–80% | conversion (A) |
| Onyeka Okongwu (ATL) | backup C | Capela out | tail (B) |
| Goga Bitadze (ORL) | backup C | Carter out | conversion (A) |
| Larry Nance Jr. | all-round F-C | P/R/A | all-round (F) |
| Yaxel Lendeborg (GSW) | rookie F | 10-24 @PHX: Horford (new), Draymond, Porziņģis all out → 36 min, 26/9/5 | role change (E), all-round (F) |
| **Tari Eason (HOU)** | F, very uneven minutes | production gated by minutes: 15p/8r is 5% at 16–22 min, **24–27% at 23+**; pts~min corr 0.54–0.63 | needs "which absences unlock his 23+ min" |

