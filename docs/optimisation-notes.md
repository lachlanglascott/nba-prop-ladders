# Optimisation notes (live lessons → changes)

Running log of what the live slates taught us and what changed. Newest first. Live record
from 4 Nov 2026 (card + triggered "if they sit" bets, 1u flat).

## 10 Nov evening: price plays section; side-aware blowout minutes; singles lead
- **Price plays (top 5)** on the slate: no-angle projection-vs-price bets, one per player, honest EV ≥ +4%, Kelly-ranked,
  0.25u, graded as `PP`. 12-15 mpg players (backup bigs) now enter this pool only (spots still need 15+).
- **Blowout minutes by side** (`blowout.CURVE_SIDE`, 69k player-games 2023-26): starters lose 2-6 min at 15+ either side;
  the *winners'* 12-18 mpg bench does not gain (+1.0 close → +0.0-0.4 at 10-25); the losers' bench (+1.2-2.8) and the
  deep bench (<12, both sides at 25+) do. The simulator draws a signed margin ~ N(−spread, 12.5).
- **Card bet leads with the best firm single** on the spot's stats; combos stay listed (user call, 10 Nov). Replay 4-9 Nov:
  same 9/28 hits, honest exp 7.9 → 8.4, P&L +7.9 → +5.9u (McDaniels' SGM win became a 20+ pts loss).
- TODO: last season's matching games only count on the same team. Jaquez without Herro at MIA 2025-26 (47 games: 30.0 min
  vs 25.5, 20+ pts 36% vs 18%) is invisible now that both are at MIL → match on the teammate, not the team.

## 10 Nov (after 9 Nov): card 0/5 (expected 1.4), IF 2/6 +2.4u; live 12/39, +11.1u, honest expected 10.8
- **Own status costs minutes:** Allen (Q knee) 19 min vs 31 projected. 2025-26, 20+ mpg players who played: Questionable
  −2.15 min vs previous 10 (all games −0.33), Probable/Available −1.1; 14% lost 8+ min (vs 9%) → `OWN_STATUS_MIN`
  (Q 1.8, P/Available 0.8; rest listings excluded). Previously only confidence was cut.
- **Strong-leg single vs combo, live:** combos 0/6 (exp 1.1, −6.0u), same-stat singles 1/6 (exp 2.0, −2.9u). Amen 9r/4a L
  with 9 reb. Proposed: bounce spots lead with the single on the bounce stat (awaiting the user's call).
- **50/50 star teammate:** Davis (55%) played, 44 pts; George had the minutes (35) but 2 ast. Watch: a role spot next to a
  coin-flip star is really a conditional.
- **Estimated role** (Thybulle, no game without Grimes) 23 vs 31 min: third miss; C grade size was right.
- **Pull bug:** raw snapshots were keyed by local date, so the AU-morning pull (US games live) was reused all day →
  refresh now refetches unless the snapshot is < 1 hour old (later same-day files get a time suffix; raw stays append-only).

## 10 Nov: per-minute volume in a new role (Dosunmu TODO) — answer: flatter, not more recent
- Tested offline (`uv run python -m nba_ladders.rate_eval`): 21k games 2024-26 where a key teammate (24+ mpg) was out,
  predicting rebound chances / potential assists / points (count given actual minutes) from his earlier games without
  that teammate, shrunk to his normal-lineup rate.
- **Shorter recency was worse** (half-life 1.5: +1-6% error). Best at every sample size (1-2, 3-6, 7+ matching games):
  **half-life 10 with no lineup discount** (−1 to −3% error on all three stats vs half-life 3 × lineup weights).
  De-inflating games by a measured per-absence lift was worse than either (lifts are noisy).
- Change: rates in matching games use `RATE_HALF = 10` and no lineup weights; **minutes keep half-life 3 and the
  position-overlap lineup weights** (not re-tested here).
- Replay 4-8 Nov: identical card (23 tips, 8 hits), honest p moved ≤0.5pp. Dosunmu's rebound-chance projection goes
  10.7 → 11.0: his 4/5-chance games are treated as noise until they persist. Keep watching; if he's under 7 chances for
  another 3-4 games, the role really has changed and the matching sample will show it.

## 9 Nov night: lineup weights for tank-season samples
- Extra absences now discount a matching game **by position overlap** (same position ×0.5, guard-wing ×0.65, wing-big ×0.75,
  guard-big ×0.9) with a **0.25 floor**, replacing the flat 0.5 per absence (Achiuwa's 15 games had been weighted 0.03-0.13).
- Last season's matching games use a **recency half-life of 8 games** (was 3, which made 15 games count like ~2).
- Achiuwa without Sabonis: 26.8 → 27.4 min. Most of his 31-min games also had Keegan Murray (a big) and Cardwell out, so the
  discount is partly justified; he played 36 on the night. Not tuned further on one game.
- Card replay 4-8 Nov (hindsight, final reports): 8/23, +7.0u, honest expected 7.0. Traore 16+ pts (W) now beats his rare 4+ reb.

## 9 Nov evening: your style → engine tweaks
- **Your style** (bets you raised): late lineup news; singles on the player's strength in a new role or trend (Acuff, Burries,
  Sharpe 3/3); uneven R/A combos (Black 2/4, +6.8u); P/R combos for bigs (Oso x2, Okongwu, Achiuwa 0/4, the points leg failed every time).
- **Lineup-news switch:** `spots --start NAME --bench NAME --out NAME` → start/bench ±3.5 min (same both ways), out = absent.
  A news-only role grades C. Conditionals inherit the news.
- **Trend category** (minutes AND chances up over the last 3) labelled in next best, weight 1.2. Replay 4-8 Nov: 0 picks cleared
  +4% (Burries was a trend but honest EV ~0%), so it rarely fires. Next-best picks are now saved and graded.
- **Weak-leg rule REJECTED:** a 35-50% leg hit-rate rule dropped Black's and Poeltl's winners (old-role history), and tonight's
  leg chances didn't separate wins from losses on 11 combos. Kept a 20% floor per leg instead, plus the strong-leg single
  shown next to every combo.
- **Picklebet in the odds feed** (`picklebet_au`, fanatics dropped to keep 10 books). Its ladders aren't confirmed yet
  (props weren't posted at check time); Picklebet SGM builder uncalibrated.
- *(fixed 9 Nov, see below)* Achiuwa without Sabonis had 15 last-season games (~31 min), but the full-lineup weighting cut them to almost
  nothing (late-season SAC had many absences) → projection 26.8 min; he played 36. Look at how lineup weights treat
  tank-season samples.

## 9 Nov (after 8 Nov): record 10/28, +13.7u, honest expected 7.9; card 2/4
- **Strength bets won, combos lost on the weaker leg:** Achiuwa 8+ reb (14) W but 10p/7r L (7 pts); Ausar 11 reb but 3 ast (7r/5a L);
  Keyonte 6r/6a one rebound short. Burries 8+ reb, Sharpe 5+ ast, Acuff 9+ ast all W. TODO: combo weak leg needs own hit rate ≥ 35%.
- **Dosunmu without Edwards is a scorer:** 1/2/4 reb (21/26/26 pts) in his last 3. TODO: weight recent rebound chances more when the role is new.
- **Bench news over-reaction:** pulled Ziaire 13+ pts; he scored 17 in 21 min. Rule: start/bench = ±3-4 min both ways; don't pull a tip
  unless the price passes its take-at.
- Paul Reed: 10 reb in 20 min straight after the "deferral, not luck" read. One game; keep watching.

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
- Dosunmu without Edwards: rebound chances 16, 16, 4, 5, 7 (8 Nov). Backtest says don't chase two games
  (10 Nov); revisit if it stays under 7 for 3-4 more.
- Minutes weighting in matching games (half-life 3, lineup weights) hasn't had the rate_eval treatment: test it the same way.
- Recalibrate the model (raw model expected 7.2 vs 6 actual) once there are ~50 live bets.
