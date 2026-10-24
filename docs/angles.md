# Angles

Each angle is a separate hypothesis with its own trigger, target market, and test. Don't blend
them: a player can be a good fit for one and a terrible fit for another.

| # | Angle | Thesis | Key metric | Market | Status |
|---|---|---|---|---|---|
| A | **Conversion** | When minutes come, production reliably follows | P(10/10 \| 25+ min) | P+R SGM at moderate lines | screened (`docs/screens/`) |
| B | **Tail / "goes huge"** | Low hit rate, but when it hits the upper rungs hit together; priced as if independent/normal | P(higher rung \| 10/10), fair $ per rung vs offered $ | P+R ladder, long odds | first look below |
| C | **Injury trigger** | Deep-bench big whose minutes jump only when the starter sits | minutes with blocker in vs out | anything, *if* in before the book reprices | screened (blocker columns) |
| D | **Foul-trouble opponent** | Opponent bigs foul a lot, so the starter sits and the backup plays | opponent big PF/36, starter early-foul rate | backup's P+R | not started; needs play-by-play |

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
