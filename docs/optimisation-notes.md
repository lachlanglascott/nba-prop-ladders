# Optimisation notes (live lessons → changes)

Running log of what the live slates taught us and what changed. Newest first. Live record
from 4 Nov 2026 (card + triggered "if they sit" bets, 1u flat).

## 8 Nov late: P/R ladder for bigs
- **Rare lines can't be card tips** (own hit rate < 15%): Sharpe 8+ ast (2/9, 0/62) reached the card because ranking only
  penalised rare lines among a player's own bets. They're now leans (filler, 0.25u).
- **Real games, bigs (6+ reb, 167 player-seasons): P and R link far more than R/A**: 12p/10r 1.14x steady / 1.36x volatile minutes,
  18p/10r 1.20x / 1.55x. The simulator has ~1.13-1.21x for Clingan (minutes sd 7.5) → real-game table used for volatile bigs.
- **But builders charge for it:** Ighodaro 10p/7r $7 = 0.75-0.80x the book's legs; 8p/7r/3a $9 = 0.82-0.86x. So P/R is about
  fairly priced on correlation, and the value has to come from the spot (minutes/role the market hasn't caught). Builder factors:
  P/R 0.78x, 3-leg 0.84x, uneven R/A 1.10x, even R/A 0.95x, long legs 0.80x. Okongwu 18p/10r $5 excluded (0.35x legs; the lines
  probably moved after Jalen Johnson was ruled out).
- P/R ladder on every big (projected 6+ reb) in a spot: 8/7 … 18/10, 12/12. Replay 4-7 Nov: +EV combos 2/13 (expected 2.2, +6.1u
  on the two Poeltl hits). On expectation; track live.

## 8 Nov evening: your method review (high-variance players, ladders, R/A)
- **Ladder tails as single bets are NOT mispriced in our favour** (two seasons, upper rungs $3-15): −13% to −21%
  blind across stat-dispersion terciles. High-dispersion far tails (2sd+) −19% vs low −29%: right direction, still negative.
- **R/A combos for guards/wings who do both (real games, 147 player-seasons): R and A land together more than independent**:
  6r/6a 1.13x steady / 1.19x volatile minutes; 7r/7a 1.18x / 1.31x. The simulator's lift (1.05-1.20) is close
  for steady players and under for volatile ones → the SGM link is now max(sim, real-game table) for combo players.
- **Builder pricing (Black slips, bets/sgm_quotes.csv):** priced off the book's own legs. Uneven R/A combos (7r/5a, 5r/7a)
  1.08-1.28x the legs (no correlation charged); 6r/6a 0.95x; 8r/8a 0.78x. Our estimates now match the slips
  (7r/5a est $7.62 vs $7.50, 6r/6a $5.22 vs $5.25). Old estimate (0.9x median) under-priced the uneven ones.
- **The SGM finder never offered uneven combos** (each leg had to be 50%+ alone) → new R/A ladder on every combo
  guard/wing spot (5/5 … 7/7) with fair, bet-at and builder estimate; card tips show the best +EV combos.
- **Rare lines rank at half weight** (own hit rate < 15%), to lean into what he's good at.
- Replay 4-7 Nov (hindsight): R/A ladder +EV combos 4/23 (expected 3.3, +1.4u at builder estimates); uneven-only 1/10.
  Too few to call. Keep logging builder quotes and grade the R/A ladder live.

## 8 Nov (after 7 Nov): record 8/24, +11.0u, honest expected 6.4; card 3/13 vs 3.6
- **Record bug:** save() flagged every spot's top bet as a card bet (7 Nov: 12 vs 5 shown). Fixed; 7 Nov saved file
  corrected. Earlier nights were unaffected (5 or fewer spots).
- **Blowouts:** 6 of the 10 card losses involved one (Camara 21 min, POR +5.5 lost by 22). TODO: test fatter blowout
  tails for role players.
- **Rare-line role bets:** Traore 4+ reb (1/8 this season) passed only through the minutes waiver; he scored 19.
  TODO: under 15% own hit rate, prefer the near-fair alternative in a stat he produces.
- Every tip now lists his other stats (best rung each, fair or better) with a Kelly stake.

- **Projections anchored to a stale lineup (Kenrich Williams).** He's played 24/17/21/31/29/19 min since Holmgren,
  J. Williams and Wallace went out, but his "normal lineup" games were October's 7/6/7. With Jaylin Williams newly out,
  only 2 games matched and the shrink pulled him to 11.5 min. Fix: teammates out 3+ straight games are the
  *established* lineup. The projection shrinks towards his games in that setup (3 pseudo-games), plus 1 pseudo-game
  of the old lineup. Kenrich is now 22.6 min / 4.9 reb (market agrees, so no bet).
  Replay of the live nights with the fix: card 4/13 (+0.9u, honest expected 3.9) vs actual 1/8 (−4.5u). It's in
  hindsight (final reports), so it's a direction check, not proof.
- **Games already tipped are excluded** (commence before the odds snapshot: DEN@IND was on the re-run card).

## 7 Nov (after 6 Nov results): record 6/19, +8.8u, honest expected 5.0
- **No 3+ rebound or assist singles (8 Nov, your call):** reb/ast singles start at 4+ (Wendell Carter 3+ ast was on the card). SGM legs can still use 3s.
- **Star filter back on (8 Nov):** you preferred second-tier players only once Harden and Mobley reached the card. Stars can still be priced on request.
- *(7 Nov, reverted)* **Star filter removed** (20+ ppg last season were skipped; Sengun on a tear wasn't even looked at).
  Tested on 6 and 7 Nov: the 6 Nov card was unchanged; 7 Nov gains Franz Wagner 8+ reb (A, Suggs out). Guard:
  a star with no lineup angle gets half weight in the next-best ranking, so Wembanyama and Sengun don't push out
  second-tier plays (Clingan, Dybantsa).
- **Bet choice ranked by raw EV favoured long prices.** Kyshawn George (if Davis sits): 8+ reb $7.75
  (+24%, 16% chance) was shown over 20+ pts $3.70 (+14%, 31%); he scored 33. Klay 19+ over 17+
  (5 Nov) was the same thing. Kelly growth rates the George pair about equal, so **show both**:
  - ranking is Kelly growth everywhere (card, IF list, next best)
  - IF list: best bet **per stat**, up to 2 per player
  - every tip has an "Also:" line with his best bet in another stat
- **Same-game stacking.** 4 CHA tips (one angle) in a +9.5 game, which they lost by 39 → max 2 card tips per
  game and max 1 from an 8.5+ underdog.
- **Estimated roles are overconfident.** No game yet without the missing teammate: live 2/11 vs model 4.5
  (honest 2.9); measured spots 4/8 vs honest 2.1 → estimated roles confidence ≤ 0.15, grade ≤ C.
- **"Never got it" guard.** 0 hits in 6+ games this season now blocks a line even if last season
  had them (Clingan 4+ ast 0/8).
- Next-best doc counts only firm tips (not leans or IF bets) towards the 3.
- Record counts a bet once even with two triggers (Poeltl: Barnes or Quickley).

## 6 Nov
- **Odds snapshot cap hid 4 of 9 games** (40 credits ÷ 7 per game) → default 100; check events-with-props
  count against the schedule every run.
- **Absences under 20 mpg were invisible** (Missi 19.5, Stewart 18.3) → 16 mpg.
- **Long absences dropped**: "new" absence required ≥50% games played (Edwards 4/9) → 30%.
- **Spots without props were silently dropped** → "waiting for prices" section with bet-at rungs.
- Pull with one season wiped the others → pull keeps unpulled seasons.

## 5 Nov
- **Will Richard**: matching games ignored other absences (Horford, Williams also out 2 Nov) → full
  lineup weighting (12+ mpg teammates; each extra absence halves the game's weight).
- **"Starting" overreaction**: priced from 24+ minute games (circular). He started, played 16, 0 reb.
  Rule: starting news for a bench player ≈ +3–4 min, wide range; never quote huge EVs on late news.
  TODO: pull starter flags from box scores so "games he started" can be measured.
- Blowouts cut fringe players' minutes (GSW +20).

## 4 Nov and before
- Guard rebounds vs extreme opponents underweighted (Dosunmu vs HOU) → OPP_W_REB_G 1.3, de-opponented samples.
- Volume-only role spots need ≥5 matching games; plausibility guard (Clingan/Gobert 4+ ast).
- Bounce baseline = half career, half this season (Coulibaly's slump was structural).

## Open questions / next
- Starter flags (box score START_POSITION) for starting-news pricing.
- Blowout weight on fringe minutes, for favourites' benches as well as underdogs.
- Dosunmu 7+ reb has lost twice with the minutes there (1 and 2 rebounds): check his rebound chances
  once 6 Nov tracking posts, in case his rebounding role without Edwards isn't what the 12/9 games implied.
- Recalibrate the model (raw model expected 7.2 vs 6 actual) once there are ~50 live bets.
