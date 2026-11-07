# Optimisation notes (live lessons → changes)

Running log of what the live slates taught us and what changed. Newest first. Live record
from 4 Nov 2026 (card + triggered "if they sit" bets, 1u flat).

## 7 Nov (after 6 Nov results): record 6/19, +8.8u, honest expected 5.0
- **Star filter removed** (20+ ppg last season were skipped; Sengun on a tear wasn't even looked at).
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
