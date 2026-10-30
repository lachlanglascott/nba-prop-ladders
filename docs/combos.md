# Combos: shape the bet to the thesis (2026-10-31)

`uv run python -m nba_ladders.combos "Precious Achiuwa" --opponent NOP --as-of 2026-10-29`

**Why:** on 2026-10-29 Achiuwa (luck reversion + SAC frontcourt out) played 41 min for 24p/9r.
The single 10+ REB alt lost; the P+R combos the thesis implied all won. When the thesis is
*minutes*, every stat that scales with minutes moves together, so combos capture it better
than one leg.

**What it does:** simulates the player (role shift, recency, luck, opponent rebounds, same as
`price`), prices every combo of points / rebounds / assists lines in a bettable range ($1.80 to
$15) where every leg is ≥25% on its own, and ranks by **lift** = P(all legs) / product of
legs, i.e. how much pricing the legs as independent would under-price it. Shown per shape (P/R,
P/A, R/A, P/R/A) so 2-leg combos aren't buried under 3-leg ones.

**Achiuwa replay (data before tip only):** top P/R combos 15p/10r ($7.30 fair vs $11.19
independent), 12p/10r ($5.11 vs $7.28), **15p/8r ($4.99 vs $6.96)** ✓, 10p/10r ($4.28 vs $5.74);
P/R/A 15p/8r/2a ($8.69 vs $15.88) ✓.

**Caveat:** lift measures how wrong *independent* pricing would be. Real SGM pricing adds some
correlation, so the edge is the gap between the book's price and `fair`, not `fair_indep`.
That needs a price feed.

The game-day sheet's **Potentials** section attaches each luck candidate's best combos.
