"""Spots: situation-first slate. Find why a player is different tonight, project from the games
that match tonight, price honestly against the market, and build bets in the user's shapes.

    uv run python -m nba_ladders.spots --date 2026-11-04          # docs/slates/<date>-spots.{md,html}
    uv run python -m nba_ladders.spots --date 2026-11-01 --grade  # score a past card against results

1. Situations (a spot needs a primary one; opposition alone is never enough):
   - role:     a rotation teammate (20+ mpg) who usually plays is out tonight (new absence)
   - backup:   the starting big is out and this player is the next big
   - emerging: same lineup, but his volume (potentials, rebound chances, minutes) jumped over
               his last 3 games versus his baseline
   - bounce:   recent volume is normal or better but conversion is far below his career rate
   - opp hole (secondary): what the opponent allows to his position in the bet's stats
2. Projection from the games that match tonight: his own games with tonight's key absences (else
   recent same-lineup games), recency-weighted and shrunk to his baseline; per-minute volume x
   career conversion (potentials -> assists, rebound chances -> rebounds); position-group opponent
   factors; minutes drawn with the spread's blowout curve. Counts are Poisson given minutes and rate
   (measured 2025-26: assists var/mean 1.07, rebounds 1.05 once minutes and rate are known).
3. Honest pricing: start from the market's own probability (median across books, de-vigged) and
   move toward the model only by a confidence factor (sample size, lineup certainty, minutes
   stability, rookie). EV shown is that honest EV, not the raw model's.
4. Bets in the user's shapes: 2-3 leg SGMs from one book's own ladder (Betr, then Sportsbet), each
   leg live on its own, plus singles $2.50-$10. Conditional bets: "if X sits".
"""

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from itertools import combinations, product

import numpy as np
import polars as pl

from nba_ladders import config
from nba_ladders.absences import position_groups
from nba_ladders.luck import K_CONV, league_conversion
from nba_ladders.luck import load as load_tracking
from nba_ladders.market import TEAM_ABBR, latest
from nba_ladders.opp_position import BOOST_C, OPP_W, angle_h
from nba_ladders.opp_position import line as opp_h_line
from nba_ladders.opp_position import tonight as opp_tonight
from nba_ladders.scenarios import positions
from nba_ladders.screens import parse_player_index
from nba_ladders.sources.injury_report import ET, player_key, report_for_date
from nba_ladders.sources.nba_stats import get_player_index
from nba_ladders.sources.odds_api import load_cached

SEASON = "22026"
ROT_MPG = 16.0  # teammates whose absence changes a lineup (was 20: missed Yves Missi 19.5, Isaiah Stewart 18.3 as starters/backup bigs)
LINEUP_MPG = 12.0  # wider net for "was the rest of the lineup the same?" (Horford 18.7, B. Williams 17.4: Will Richard 2 Nov)
EXTRA_OUT_W = 0.5  # a matching game counts this much less per extra teammate missing then who plays tonight
KEY_MPG = 24.0  # a "key" absence for role spots
CAND_MPG = 15.0  # players considered (recent minutes)
PRICE_MPG = 12.0  # 12-15 min players (backup bigs: Drummond 13) are priced too, for the price-play section only
PRICE_N = 5  # price plays listed on the slate
STAR_PPG = 20.0  # last season: stars are left out (sharply priced; tried back in 7 Nov, user preferred them out 8 Nov)
OUT_P = 0.70  # P(sits) at which a teammate counts as out tonight
# Minutes lost when he is on the report himself and plays, beyond normal drift (2025-26, 20+ mpg, vs his previous 10:
# Questionable -2.15 / Probable -1.1 / Available -1.14 against -0.33 for all games; 14% lose 8+ min vs 9%).
OWN_STATUS_MIN = {"Questionable": 1.8, "Probable": 0.8, "Available": 0.8}


def own_status_cut(st: str, reason: str = "") -> float:
    if "rest" in (reason or "").lower():  # a rest risk who plays is a normal game (measured on injury listings only)
        return 0.0
    return next((v for k, v in OWN_STATUS_MIN.items() if (st or "").startswith(k)), 0.0)


SAFE_P = 0.60  # the player himself: P(sits) below this (props are void if he doesn't play; uncertainty costs confidence)
# Market calibration: actual hit rate / implied probability by price, 664k Sportsbet ladder rungs 2024-26
# (alt overs are priced short, more so at longer odds). Used to turn a price into the market's real probability.
MKT_PRICE = np.array([1.24, 1.74, 2.24, 2.76, 3.44, 4.5, 5.62, 7.34, 9.12, 11.79])
MKT_RATIO = np.array([0.934, 0.929, 0.899, 0.889, 0.856, 0.82, 0.789, 0.792, 0.769, 0.674])


def market_ratio(price: float) -> float:
    return float(np.interp(np.log(price), np.log(MKT_PRICE), MKT_RATIO))
CONF_MIN, CONF_MAX = 0.08, 0.35
SGM_FACTOR = 0.90  # fallback: builder ≈ this x product of the AU-median legs (when the book's own legs aren't listed)
# Builder ≈ factor x product of THE SAME BOOK's legs, by shape (Black 7 Nov slips, bets/sgm_quotes.csv; provisional):
# uneven (one leg <= $2, one >= $3): 1.08-1.28x (no correlation charged); even mid legs 0.95x; long legs (all >= $5) 0.78x
# P/R: builders DO charge the correlation (Ighodaro 10p/7r $7 = 0.75-0.80x legs; 8p/7r/3a $9 = 0.82-0.86x).
def builder_factor(prices: list[float], stats: set | None = None) -> float:
    if stats == {"pts", "reb"}:
        return 0.78
    if len(prices) >= 3:
        return 0.84
    if min(prices) <= 2.0 and max(prices) >= 3.0:
        return 1.10
    if min(prices) >= 5.0:
        return 0.80
    return 0.95


# R/A link in real games, combo guards/wings (3.5-8 reb and ast; 147 player-seasons 2023-26): joint / product of marginals.
# Volatile-minutes players run higher than the simulator (7r/7a 1.31 vs sim ~1.16-1.20).
RA_LIFT = {"steady": {(5, 5): 1.04, (6, 6): 1.13, (7, 5): 1.08, (5, 7): 1.10, (7, 7): 1.18, (6, 5): 1.08, (5, 6): 1.07},
           "volatile": {(5, 5): 1.12, (6, 6): 1.19, (7, 5): 1.21, (5, 7): 1.16, (7, 7): 1.31, (6, 5): 1.15, (5, 6): 1.15}}
VOLATILE_MIN_SD = 5.5  # minutes sd (his matching games) at which a player counts as volatile


# P/R link in real games, bigs (6+ reb, 6-22 pts; 167 player-seasons 2023-26), keyed (pts, reb). Far stronger
# than R/A when minutes swing (12p/10r 1.36, 18p/10r 1.55): the simulator has ~1.13-1.21 for Clingan.
PR_LIFT = {"steady": {(8, 7): 1.05, (10, 7): 1.07, (12, 7): 1.08, (15, 7): 1.10, (8, 10): 1.07, (10, 10): 1.11,
                      (12, 10): 1.14, (15, 10): 1.17, (18, 10): 1.20, (12, 12): 1.17, (20, 12): 1.32},
           "volatile": {(8, 7): 1.15, (10, 7): 1.19, (12, 7): 1.23, (15, 7): 1.28, (8, 10): 1.22, (10, 10): 1.29,
                        (12, 10): 1.36, (15, 10): 1.46, (18, 10): 1.55, (12, 12): 1.43, (20, 12): 1.85}}


def ra_lift(a: dict, legs) -> float | None:
    """Real-game lift for a 2-leg combo: R/A for combo guards/wings, P/R for bigs (nearest tabled combo)."""
    d = dict(legs)
    vol = "volatile" if a.get("min_sd", 0) >= VOLATILE_MIN_SD else "steady"
    if set(d) == {"reb", "ast"} and a.get("combo_ra"):
        r_, a_ = min(max(d["reb"], 5), 7), min(max(d["ast"], 5), 7)
        key = min(RA_LIFT["steady"], key=lambda k: abs(k[0] - r_) + abs(k[1] - a_))
        return RA_LIFT[vol][key]
    if set(d) == {"pts", "reb"} and a.get("big"):
        p_, r_ = min(max(d["pts"], 8), 20), min(max(d["reb"], 7), 12)
        key = min(PR_LIFT["steady"], key=lambda k: abs(k[0] - p_) / 2 + abs(k[1] - r_))
        return PR_LIFT[vol][key]
    return None
LEG_MIN_P = 0.50  # each SGM leg live on its own
SGM_MIN_EV = 0.08
CARD_N = 5
PER_GAME = 2  # card tips per game (6 Nov: 4 CHA tips, CHA lost by 39, all four lost)
DOG_CAP, DOG_SPREAD = 1, 8.5  # at most this many from a team that's an underdog by this much (blowout = minutes cut)
STAR_NO_ANGLE = 0.5  # next-best weight for a star with no lineup angle (Wembanyama/Sengun pushed out Clingan/Dybantsa)
MIN_TIPS = 3  # fewer card tips than this: write a "next best" doc to make it up
NEXT_FLOOR = -0.08  # next-best pool: closest-to-fair bets down to this honest EV
LEG_MIN = {"pts": 8, "reb": 3, "ast": 3}
SGM_BOOKS = ("betr_au", "sportsbet", "picklebet_au")  # the user's SGM books (Picklebet builder not calibrated yet)
SINGLE_RANGE = (2.5, 10.0)
SINGLE_MIN_LINE = 4  # reb/ast singles start at 4+
SGM_RANGE = (3.5, 15.0)
FREED_CAP = 6.0  # minutes: the rough share-of-freed-minutes estimate when there's no game without them
# Opponent x position in the projection: opp_position.OPP_W (every stat, every group), replacing pts x1.5 / guards' reb x1.3
KEY_SHARE = 0.3  # (was 0.5: dropped Edwards after 5 missed of 9) an absence is 'new' if he played this share of this season's games (else his teammates' roles already reflect it)
PRIMARY = {"role", "bounce"}  # spot types that stand on their own
EST_CONF = 0.15  # confidence cap when the role is estimated, not measured
ROLE_MIN, ROLE_VOL = 2.0, 1.15
ROLE_VOL_GAMES = 5  # a volume-only role spot (no extra minutes) needs this many matching games (Clingan, Gobert)  # a role spot needs +2 min or +15% volume vs his normal-lineup games
BOUNCE_GAP, BOUNCE_VOL = 0.18, 6.0  # conversion this far under career, on 6+ chances a game, over 5+ games
OUTLIER = 1.35  # a book's price this far above the market's fair = stale or wrong line: check before betting
MIN_EV = 0.04  # honest EV to list a bet
N_SIM = 40_000
RNG = np.random.default_rng(11)


# ---------------------------------------------------------------- data as at the day
def _frames(day: date):
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(pl.col("game_date") < day)
    trk = load_tracking().filter(pl.col("game_date") < day)
    return pg, trk


def _report(day: date, pg: pl.DataFrame) -> pl.DataFrame:
    from nba_ladders.candidates import add_b2b_rest, add_carry_forward

    rep = report_for_date(day).filter(pl.col("game_date") == day).with_columns(
        pl.col("team_name").replace_strict({k.replace(" ", ""): v for k, v in TEAM_ABBR.items()}, default=None).alias("team"))
    rep = add_b2b_rest(rep, pg, day)
    return add_carry_forward(rep, pg, day)


def _p_sit(row: dict) -> float:
    from nba_ladders.candidates import p_sits

    return p_sits(row)


def _odds(day: date) -> pl.DataFrame:
    o = latest(load_cached("live"))
    lo = datetime(day.year, day.month, day.day, 6, tzinfo=ET)
    return o.filter(pl.col("commence_time").map_elements(lambda s: lo <= datetime.fromisoformat(s) < lo + timedelta(days=1),
                                                         return_dtype=pl.Boolean),
                    pl.col("side") == "Over", pl.col("threshold").is_not_null())


def _spreads(day: date) -> dict:
    from nba_ladders.sources.odds_api import RAW_DIR, parse_spreads

    best = {}
    for f in sorted((RAW_DIR / "spreads").glob("*.json")):
        for r in parse_spreads(json.loads(f.read_text())).iter_rows(named=True):
            d = datetime.fromisoformat(r["commence_time"]).astimezone(ET).date()
            if d == day:
                best[TEAM_ABBR.get(r["team_name"])] = r["spread"]
    return best


# ---------------------------------------------------------------- per-player history with lineups
def _game_table(pg: pl.DataFrame, trk: pl.DataFrame, team_now: dict, season: str = SEASON) -> pl.DataFrame:
    """A season's player-games with tracking and the set of absent rotation teammates."""
    cur = pg.filter(pl.col("season_id") == season)
    mp = (cur.group_by("player_id", "team").agg(pl.col("minutes").mean().alias("mpg"), pl.len().alias("gp"))
          .filter(pl.col("gp") >= 2))
    games = cur.select("team", "game_id").unique()
    played = cur.filter(pl.col("minutes") > 0).select("team", "game_id", "player_id")

    def missing(min_mpg, col):
        return (games.join(mp.filter(pl.col("mpg") >= min_mpg).select("team", "player_id"), on="team")
                .join(played, on=["team", "game_id", "player_id"], how="anti")
                .group_by("team", "game_id").agg(pl.col("player_id").alias(col)))
    t = trk.filter(pl.col("season_id") == season).select("player_id", "game_id", "potential_ast", "reb_chances")
    return (cur.join(t, on=["player_id", "game_id"], how="left")
            .join(missing(ROT_MPG, "absent"), on=["team", "game_id"], how="left")
            .join(missing(LINEUP_MPG, "absent_wide"), on=["team", "game_id"], how="left")
            .with_columns(pl.col("absent").fill_null([]), pl.col("absent_wide").fill_null([]))
            .sort("player_id", "game_date"))


def _conversions(trk: pl.DataFrame) -> tuple[dict, dict]:
    lg = league_conversion(trk.filter(pl.col("season_id") != SEASON))
    agg = trk.group_by("player_id").agg(pl.col("ast").sum(), pl.col("potential_ast").sum(),
                                        pl.col("reb").sum(), pl.col("reb_chances").sum())
    ca = {r[0]: (r[1] + K_CONV["ast"] * lg["ast"]) / (r[2] + K_CONV["ast"]) for r in agg.select("player_id", "ast", "potential_ast").iter_rows()}
    cr = {r[0]: (r[1] + K_CONV["reb"] * lg["reb"]) / (r[2] + K_CONV["reb"]) for r in agg.select("player_id", "reb", "reb_chances").iter_rows()}
    return ca, cr


# ---------------------------------------------------------------- situations + projection
def _weights(n: int, half: float = 3.0) -> np.ndarray:
    return 0.5 ** (np.arange(n)[::-1] / half)  # most recent last


LAST_HALF = 8.0  # recency half-life (games) inside last season's matching games: 3 made 15 games count like ~2
# Per-minute volume (rebound chances, potentials, points) in matching games: almost flat recency and no lineup
# discount. rate_eval on 21k new-role games 2024-26: half-life 10 with no lineup weights cut count error 1-3% on all
# three stats at every sample size vs half-life 3 x lineup weights; a shorter half-life (1.5) was worse. Dosunmu's
# 4-5 rebound chances after 16/16 (Nov 2026) is noise until it persists. Minutes keep half-life 3 and lineup weights.
RATE_HALF = 10.0


def _season_weights(season_ids: np.ndarray, half: float = 3.0) -> np.ndarray:
    """Recency weights for a matching sample (oldest first): this season half-life `half` games; last season counts
    half, with a slower half-life so a long run of matching games (Achiuwa without Sabonis, 15) isn't reduced to two."""
    this = season_ids == SEASON
    w = np.empty(len(season_ids))
    w[this] = _weights(int(this.sum()), half)
    w[~this] = 0.5 * _weights(int((~this).sum()), max(half, LAST_HALF))
    return w


def _per36(df: pl.DataFrame, col: str) -> float | None:
    d = df.drop_nulls(col)
    return float(d[col].sum() / d["minutes"].sum() * 36) if d.height and d["minutes"].sum() > 0 else None


def _wmean(df: pl.DataFrame, col: str, w: np.ndarray) -> float:
    return float(np.sum(df[col].to_numpy() * w) / np.sum(w))


NEWS_MIN = 3.5  # minutes for start/bench news (a modest bump, not starter minutes)
ESTABLISHED = 3  # a teammate out this many straight games is part of the current lineup
K_OLD = 1.0  # pseudo-games of his old-lineup minutes kept in the projection (keeps 4-game samples honest)


def _streak(absent_sets: list, m: int) -> int:
    """How many of his most recent games teammate m has missed in a row."""
    n = 0
    for a in reversed(absent_sets):
        if m not in a:
            break
        n += 1
    return n


# How much an extra absence (missing then, plays tonight) inflates his minutes, by position pair: a missing big frees a
# big's minutes, a missing guard mostly a guard's. Achiuwa 8 Nov: his 15 games without Sabonis (late 2025-26, LaVine/Monk/
# Hunter/Murray all out) were weighted 0.03-0.13 by a flat 0.5 per absence → projection 26.8 min; he played 36.
EXTRA_OUT_BY_POS = {frozenset("G"): 0.5, frozenset("W"): 0.5, frozenset("B"): 0.5, frozenset("GW"): 0.65,
                    frozenset("WB"): 0.75, frozenset("GB"): 0.9}
LW_FLOOR = 0.25  # a matching game never counts for less than this
BIG_R36 = 11.0  # reb per 36 that make a listed forward a big for "the big ahead of him" (Diabate F 11.4, Giannis, Beringer)


def _lineup_weights(sample: pl.DataFrame, pid: int, tonight_in: set | None, id_name: dict) -> pl.DataFrame:
    """Each matching game's weight and who else was missing then but plays tonight (position-aware)."""
    extra = [sorted(set(a) & (tonight_in or set()) - {pid}) for a in sample["absent_wide"].to_list()]
    me = id_name.get(("grp", pid), "W")
    my_mpg = id_name.get(("mpg", pid), 0.0)
    # a big's games without the big ahead of him (more minutes, plays tonight) are a different role, not a
    # discounted match: Kalkbrenner 12 Nov, his only matching game was 35 min / 15 reb with Diabate out
    # (a "big" here includes forwards who rebound like centres: Diabate is listed F, 11.4 reb per 36)
    big = lambda m: id_name.get(("grp", m), "W") == "B" or id_name.get(("r36", m), 0.0) >= BIG_R36
    ahead = lambda m: me == "B" and big(m) and id_name.get(("mpg", m), 0.0) > my_mpg
    lw = [0.0 if any(ahead(m) for m in e) else
          max(LW_FLOOR, float(np.prod([EXTRA_OUT_BY_POS.get(frozenset(me + id_name.get(("grp", m), "W")), EXTRA_OUT_W) for m in e])))
          for e in extra]
    return sample.with_columns(pl.Series("lw", lw, dtype=pl.Float64),
                               pl.Series("also_out", [", ".join(id_name.get(m, str(m)) for m in e) for e in extra], dtype=pl.Utf8))


def analyse(pid: int, name: str, team: str, opp: str, g: pl.DataFrame, last: pl.DataFrame, tonight_out: set,
            tonight_unc: dict, id_name: dict, ca: float, cr: float, opp_f: dict, spread: float | None,
            g_last: pl.DataFrame | None = None, opp_tab: dict | None = None, grp_me: str = "W",
            tonight_in: set | None = None, usage_fn=None) -> dict | None:
    """Situations, comparable games and the projection for one player. g = his games this season (current team).
    tonight_in = teammates (12+ min) expected to play tonight: a matching game where some of them were also
    missing is a bigger role than tonight's, so it counts less (EXTRA_OUT_W per extra absence)."""
    if g.height < 2:
        return None
    absent_sets = [set(a) for a in g["absent"].to_list()]
    played_share = {m: sum(m not in a for a in absent_sets) / len(absent_sets) for m in tonight_out}
    key_new = {m for m in tonight_out if played_share[m] >= KEY_SHARE}
    sits = []
    base_mask = np.array([not (key_new & a) for a in absent_sets])
    if key_new:
        comp_mask = np.array([key_new <= a for a in absent_sets])
    else:
        usual = set(tonight_out)
        comp_mask = np.array([a == usual for a in absent_sets])
        if comp_mask.sum() < 2:
            comp_mask = np.array([len(a ^ usual) <= 1 for a in absent_sets])
    comp, base = g.filter(pl.Series(comp_mask)), g.filter(pl.Series(base_mask))
    # baseline volume: this season's normal-lineup games + last season (3 pseudo-games)
    def per36_blend(df, col, prior):
        v = _per36(df, col)
        n = df.drop_nulls(col).height
        if v is None:
            return prior
        return v if prior is None else (v * n + prior * 3) / (n + 3)
    last_pot = _per36(last, "potential_ast") if last.height >= 10 else None
    last_rc = _per36(last, "reb_chances") if last.height >= 10 else None
    last_pts = _per36(last, "pts") if last.height >= 10 else None
    base_min = float(base["minutes"].mean()) if base.height else float(g["minutes"].mean())
    b_pot, b_rc, b_pts = (per36_blend(base, "potential_ast", last_pot), per36_blend(base, "reb_chances", last_rc),
                          per36_blend(base, "pts", last_pts))
    # ---- sample that matches tonight
    # last season on the same team, with the same key players out (White without Tatum, 2025-26)
    last_comp = None
    if key_new and comp.height < 3 and g_last is not None and g_last.height:
        la = [set(a) for a in g_last["absent"].to_list()]
        lm = np.array([key_new <= a for a in la])
        if lm.sum() >= 3:
            last_comp = g_last.filter(pl.Series(lm)).tail(15)
    if key_new and (comp.height or last_comp is not None):
        parts = [comp] + ([last_comp] if last_comp is not None else [])
        sample = pl.concat([x.select(comp.columns) for x in parts], how="vertical_relaxed").sort("game_date")
        sample = _lineup_weights(sample, pid, tonight_in, id_name)
        sample = sample.filter(pl.col("lw") > 0) if sample["lw"].max() > 0 else None
        why = "games with " + ", ".join(sorted(id_name.get(m, str(m)) for m in key_new)) + " out" + (
            f" ({comp.height} this season, {last_comp.height} last season)" if last_comp is not None else "")
    elif key_new:
        sample, why = None, ""
    else:
        sample, why = _lineup_weights(comp.tail(6), pid, tonight_in, id_name), "recent games with tonight's lineup"
        sample = sample.filter(pl.col("lw") > 0)
    # the role he's in now: teammates out 3+ straight games are the established lineup, so the projection
    # shrinks toward his games in that setup, not the old lineup (Kenrich Williams 7 Nov: 7/6/7 min in
    # October with Holmgren/J. Williams/Wallace playing vs 24-31 since; shrinking to 6.7 gave 11.5)
    p_min, p_pot, p_rc, p_pts = base_min, b_pot, b_rc, b_pts
    est = {m for m in key_new if _streak(absent_sets, m) >= ESTABLISHED}
    if est:
        cur_role = g.filter(pl.Series([est <= a and not ((key_new - est) & a) for a in absent_sets])).tail(8)
        if cur_role.height >= 2:
            p_min = float(cur_role["minutes"].mean())
            p_pot, p_rc, p_pts = (per36_blend(cur_role, "potential_ast", last_pot), per36_blend(cur_role, "reb_chances", last_rc),
                                  per36_blend(cur_role, "pts", last_pts))
    # usage: what this player picks up per minute when tonight's new absentees sit (usage.py: league prior +
    # his own past games without each of them, any season/lineup), scaling the prior the matching games blend
    # with (so it matters most when they're few). Established absences are already in cur_role.
    new_out, usage_note = key_new - est, ""
    if new_out and usage_fn is not None:
        um = usage_fn(pid, new_out)["mult"]
        p_pts = p_pts * um["pts"] if p_pts is not None else None
        p_pot = p_pot * um["ast"] if p_pot is not None else None
        p_rc = p_rc * um["reb"] if p_rc is not None else None
        usage_note = "usage without them: " + ", ".join(f"{s} {(um[s] - 1) * 100:+.0f}%" for s in ("pts", "ast", "reb"))
    proj_min, pot36, rc36, pts36, n_comp = p_min, p_pot, p_rc, p_pts, 0
    if sample is not None and sample.height:
        lw = sample["lw"].to_numpy()
        w = _season_weights(sample["season_id"].to_numpy()) * lw
        this_s = sample["season_id"].to_numpy() == SEASON
        n_comp = round(float(lw[this_s].sum() + 0.5 * lw[~this_s].sum()))  # effective games: extra absences count less
        k = 3.0  # pseudo-games of baseline
        s_min = _wmean(sample, "minutes", w)
        proj_min = (s_min * w.sum() + p_min * k + base_min * K_OLD) / (w.sum() + k + K_OLD)
        stat_of = {"potential_ast": "ast", "reb_chances": "reb", "pts": "pts"}

        def blend36(col, prior):
            d = sample.drop_nulls(col)
            if not d.height:
                return prior
            ww = _season_weights(d["season_id"].to_numpy(), RATE_HALF)  # no lineup discount for rates (rate_eval)
            # remove each matching game's opponent effect, so tonight's opponent is applied to a neutral rate
            of = np.array([(opp_tab or {}).get(o, {}).get(stat_of[col], 1.0) for o in d["opponent"].to_list()]) ** OPP_W[grp_me][stat_of[col]]
            v = float(np.sum(d[col].to_numpy() / of * ww) / np.sum(d["minutes"].to_numpy() * ww) * 36)
            return v if prior is None else (v * ww.sum() + prior * k) / (ww.sum() + k)
        pot36, rc36, pts36 = blend36("potential_ast", p_pot), blend36("reb_chances", p_rc), blend36("pts", p_pts)
    elif key_new:  # no game yet without them: a share of the freed minutes, mostly to the same position
        mine = id_name.get(("grp", pid), "W")
        freed = sum(id_name.get(("mpg", m), 30.0) * (0.25 if id_name.get(("grp", m), "W") == mine else 0.06) for m in key_new)
        proj_min = min(base_min + min(freed, FREED_CAP), 40.0)
        why = "no game yet with this lineup (estimated)"
    # ---- situations
    vol_up = max((pot36 or 0) / b_pot if b_pot else 1, (rc36 or 0) / b_rc if b_rc else 1)
    if key_new and (proj_min - base_min >= ROLE_MIN or (vol_up >= ROLE_VOL and n_comp >= ROLE_VOL_GAMES)):
        names = ", ".join(sorted(id_name.get(m, str(m)) for m in key_new))
        d_min = proj_min - base_min
        sits.append({"kind": "role", "text": f"{names} out (usually plays): {d_min:+.1f} min vs his normal-lineup games"
                     + (f" ({n_comp} games without them)" if n_comp else " (no game without them yet)"), "strength": max(d_min, 0) / 3 + max(vol_up - 1, 0) * 2})
    # same-lineup games only: nobody missing then who plays tonight (an absence boost isn't a new role)
    same = g.filter(pl.Series([a <= set(tonight_out) for a in absent_sets]))
    recent = same.tail(3)
    prior = same.head(same.height - 3) if same.height > 3 else None
    if not key_new and prior is not None and prior.height >= 2:
        for col, lab in (("potential_ast", "potential assists"), ("reb_chances", "rebound chances")):
            r36, p36 = _per36(recent, col), per36_blend(prior, col, last_pot if col == "potential_ast" else last_rc)
            if r36 and p36 and r36 >= 1.25 * p36 and r36 - p36 >= 2.5:
                sits.append({"kind": "emerging", "stat": "ast" if col == "potential_ast" else "reb",
                             "text": f"{lab} per 36 up to {r36:.1f} over his last 3 (from {p36:.1f}), same lineup",
                             "strength": r36 / p36 - 1})
        rm, pm = float(recent["minutes"].mean()), float(prior["minutes"].mean())
        if rm - pm >= 4:
            sits.append({"kind": "emerging", "stat": "min", "text": f"minutes up to {rm:.0f} over his last 3 (from {pm:.0f})",
                         "strength": (rm - pm) / 8})
    r5 = g.tail(6).drop_nulls(["potential_ast", "reb_chances"])
    if r5.height >= 5:
        gs = g.drop_nulls(["potential_ast", "reb_chances"])
        for stat, ch, career, lab in (("ast", "potential_ast", ca, "assists"), ("reb", "reb_chances", cr, "rebounds")):
            now = r5[stat].sum() / max(r5[ch].sum(), 1)
            # baseline = half career, half this season: a slump all season (new teammates taking the boards,
            # e.g. Coulibaly with Davis/Sarr/Dybantsa) is structural, not bad luck
            season = gs[stat].sum() / max(gs[ch].sum(), 1)
            conv = 0.5 * career + 0.5 * season
            vol36 = _per36(r5, ch)
            basev = b_pot if ch == "potential_ast" else b_rc
            if conv and now <= conv - BOUNCE_GAP and vol36 and basev and vol36 >= 0.9 * basev and r5[ch].mean() >= BOUNCE_VOL:
                sits.append({"kind": "bounce", "stat": stat,
                             "text": f"{lab}: converting {now:.0%} of his chances over the last {r5.height} (career {career:.0%}, "
                                     f"this season {season:.0%}) on normal volume",
                             "strength": (conv - now) * 3})
    # ---- projection (means): per-minute volume x career conversion x opponent (per-minute vs own rate)
    pot36 = pot36 if pot36 is not None else (float(g["ast"].sum() / g["minutes"].sum() * 36) / max(ca, 0.3))
    rc36 = rc36 if rc36 is not None else (float(g["reb"].sum() / g["minutes"].sum() * 36) / max(cr, 0.3))
    pts36 = pts36 if pts36 is not None else float(g["pts"].sum() / g["minutes"].sum() * 36)
    ow = OPP_W[grp_me]
    rate = {"ast": pot36 * ca / 36 * opp_f.get("ast", 1.0) ** ow["ast"], "reb": rc36 * cr / 36 * opp_f.get("reb", 1.0) ** ow["reb"],
            "pts": pts36 / 36 * opp_f.get("pts", 1.0) ** ow["pts"]}
    src = sample if sample is not None and sample.height >= 3 else g.tail(8)
    min_sd = max(4.0, float(src["minutes"].std() or 0.0))
    hist = pl.concat([g.select("season_id", "pts", "reb", "ast"), last.select("season_id", "pts", "reb", "ast")]
                     if last.height else [g.select("season_id", "pts", "reb", "ast")], how="vertical_relaxed")
    return {"pid": pid, "name": name, "team": team, "opp": opp, "sits": sits, "hist": hist, "opp_tab": opp_tab or {},
            "opp_w": OPP_W[grp_me], "key_new": key_new, "n_comp": n_comp,
            "why_sample": " · ".join(x for x in (why, usage_note) if x), "sample": sample, "base_min": base_min, "proj_min": proj_min, "min_sd": min_sd,
            "rate": rate, "pot36": pot36, "rc36": rc36, "ca": ca, "cr": cr, "spread": spread, "opp_f": opp_f,
            "games": g.height, "last_games": last.height, "last5": g.tail(5)}


def opp_comps(cur: pl.DataFrame, grp: dict, opp: str, g: str, n: int = 5) -> dict:
    """What players in the same position did against this opponent in its last n games, vs their own averages."""
    top = 1 if g == "B" else 2
    d = cur.filter(pl.col("opponent") == opp, pl.col("minutes") >= 15).with_columns(
        pl.col("player_id").replace_strict(grp, default="W").alias("grp")).filter(pl.col("grp") == g)
    if d.is_empty():
        return {}
    avg = cur.filter(pl.col("minutes") >= 15).group_by("player_id").agg(*[pl.col(s).mean().alias(f"avg_{s}") for s in ("pts", "reb", "ast")])
    games = d.select("game_id", "game_date").unique().sort("game_date").tail(n)["game_id"].to_list()
    rows = (d.filter(pl.col("game_id").is_in(games)).sort("minutes", descending=True).group_by("game_id", maintain_order=True).head(top)
            .join(avg, on="player_id").sort("game_date", descending=True))
    diff = {s: float((rows[s] - rows[f"avg_{s}"]).mean()) for s in ("pts", "reb", "ast")} if rows.height else {}
    return {"rows": rows.select("game_date", "team", "player_name", "minutes", "pts", "reb", "ast",
                                *[(pl.col(s) - pl.col(f"avg_{s}")).round(1).alias(f"d_{s}") for s in ("pts", "reb", "ast")]).rows(),
            "diff": diff, "label": {"G": "guards", "W": "wings", "B": "starting bigs"}[g]}


def simulate(a: dict, n: int = N_SIM) -> dict:
    """Minutes (with the spread's blowout curve) then Poisson counts at the projected rates."""
    from nba_ladders.blowout import SD_MARGIN, side_base, side_shift
    from nba_ladders.sim import draw_points

    m = RNG.normal(a["proj_min"], a["min_sd"], n)
    if a["spread"] is not None:  # margin ~ N(-spread, 12.5); shift by role and by winning/losing side (blowout.CURVE_SIDE)
        margin = RNG.normal(-a["spread"], SD_MARGIN, n)
        m = m + side_shift(margin, a["proj_min"]) - side_base(a["proj_min"])
    m = np.clip(m, 0, 48)
    r = a["rate"]
    return {"minutes": m, "ast": RNG.poisson(r["ast"] * m), "reb": RNG.poisson(r["reb"] * m),
            "pts": draw_points(RNG, r["pts"] * m)}


def confidence(a: dict, unc_team: float, filed: bool) -> float:
    """How far to move from the market toward the model. Replay of 3 slates (167 bets): the model's
    disagreement was worth ~15-20% weight (two-season backtest agreed: small); capped by spot type."""
    c = 0.18
    if a["n_comp"] >= 3:
        c += 0.10
    elif a["n_comp"] == 0 and a["key_new"]:
        c -= 0.05
    c += 0.04 if filed else -0.04
    c -= 0.06 * min(unc_team, 1.0)  # uncertain teammates tonight
    if a["last_games"] < 10 and a["games"] < 8:
        c -= 0.06  # rookie / tiny sample
    if a["min_sd"] > 7:
        c -= 0.04
    kinds = {x["kind"] for x in a["sits"]}
    cap = CONF_MAX if "role" in kinds and a["n_comp"] >= 3 else 0.30
    if a["key_new"] and a["n_comp"] == 0:
        cap = EST_CONF  # minutes guessed (no game without them yet): live 2/11 vs model 4.5 expected, 6 Nov
    return float(np.clip(c, CONF_MIN, cap))


# ---------------------------------------------------------------- pricing and bets
AU = ("sportsbet", "tab", "dabble_au", "betr_au", "neds", "ladbrokes_au", "pointsbetau", "picklebet_au")


def market_probs(odds: pl.DataFrame, name: str) -> tuple[dict, dict]:
    """(stat, k) -> de-vigged median implied probability; and {book: {(stat, k): price}} incl. 'au_median'."""
    k = player_key(name)
    o = odds.filter(pl.col("player").map_elements(player_key, return_dtype=pl.String) == k,
                    pl.col("stat").is_in(["pts", "reb", "ast", "pts+reb", "pts+ast", "reb+ast"]))
    med = o.group_by("stat", "threshold").agg(pl.col("price").median().alias("px"))
    mk = {(r[0], r[1]): min(market_ratio(r[2]) / r[2], 0.99) for r in med.iter_rows()}
    books = {}
    for bk, st, th, pr in o.select("bookmaker", "stat", "threshold", "price").iter_rows():
        books.setdefault(bk, {})[(st, th)] = pr
    books["au_median"] = {(r[0], r[1]): r[2] for r in o.filter(pl.col("bookmaker").is_in(list(AU))).group_by("stat", "threshold")
                          .agg(pl.col("price").median()).iter_rows()}
    return mk, books


def _p(sim: dict, legs) -> float:
    hit = np.ones(len(sim["pts"]), dtype=bool)
    for s, k in legs:
        hit &= sum(sim[x] for x in s.split("+")) >= k
    return float(hit.mean())


PLAUSIBLE = 0.10  # he must have hit the line in 10%+ of games over two seasons (this season x2)...
BIG_ROLE_MIN = 4.0  # ...unless tonight's role adds 4+ minutes
HIST_W = 0.4  # weight of his own hit rate on the line (this season counts double) in the model probability


def hit_rate(a: dict, legs) -> tuple[float | None, str]:
    """His own history on these legs: weighted hit rate and a 'hit 2/6 this season, 16/77 last' label."""
    h = a.get("hist")
    if h is None or h.is_empty():
        return None, ""
    hit = pl.lit(True)
    for s, k in legs:
        hit = hit & (pl.sum_horizontal([pl.col(x) for x in s.split("+")]) >= k)
    d = h.with_columns(hit.alias("hit"))
    cur, prev = d.filter(pl.col("season_id") == SEASON), d.filter(pl.col("season_id") != SEASON)
    n = 2 * cur.height + prev.height
    if n == 0:
        return None, ""
    rate = (2 * cur["hit"].sum() + prev["hit"].sum()) / n
    lab = f"hit {cur['hit'].sum()}/{cur.height} this season" + (f", {prev['hit'].sum()}/{prev.height} last" if prev.height else "")
    return float(rate), lab


def anchored(a: dict, sim: dict, legs) -> tuple[float, str]:
    """Model probability blended with his own hit rate. For a lineup role change his old games
    describe the old role, so the anchor is the games that match tonight (3+), or none."""
    pm = _p(sim, legs)
    hr, lab = hit_rate(a, legs)
    if any(x["kind"] == "role" for x in a["sits"]):
        s = a.get("sample")
        if s is not None and s.height >= 3:
            # each matching game rescaled to tonight's opponent: value x (tonight's factor / that game's factor)
            tab, tonight = a.get("opp_tab", {}), a.get("opp_f", {})
            w = a.get("opp_w", OPP_W["W"])
            hits = np.ones(s.height, dtype=bool)
            for st, k in legs:
                tot = np.zeros(s.height)
                for x in st.split("+"):
                    game_f = np.array([tab.get(o, {}).get(x, 1.0) for o in s["opponent"].to_list()])
                    tot += s[x].to_numpy() * (tonight.get(x, 1.0) / game_f) ** w[x]
                hits &= tot >= k - 0.5  # counts are whole numbers: within half a rebound/assist counts as reaching k
            n_hit = int(hits.sum())
            lw = s["lw"].to_numpy() if "lw" in s.columns else np.ones(s.height)
            rate = float((hits * lw).sum() / lw.sum())  # games with extra teammates missing count less
            return (1 - HIST_W) * pm + HIST_W * rate, f"hit {n_hit}/{s.height} in games like tonight · {lab}"
        return pm, f"{lab} (old role)" if lab else ""
    return (pm if hr is None else (1 - HIST_W) * pm + HIST_W * hr), lab


def plausible(a: dict, legs) -> bool:
    """'He's never got 4': skip lines he almost never reaches, unless his minutes jump."""
    if a["proj_min"] - a["base_min"] >= BIG_ROLE_MIN:
        return True
    hr, _ = hit_rate(a, legs)
    h = a.get("hist")
    if h is not None and not h.is_empty():  # 0 for 6+ this season: last season doesn't rescue it (Clingan 4+ ast: 0/8, 16/77)
        cur = h.filter(pl.col("season_id") == SEASON)
        if cur.height >= 6:
            hit = pl.lit(True)
            for s, k in legs:
                hit = hit & (pl.sum_horizontal([pl.col(x) for x in s.split("+")]) >= k)
            if not cur.select(hit.alias("h"))["h"].any():
                return False
    return hr is None or hr >= PLAUSIBLE


def growth(b: dict) -> float:
    """Kelly growth per bet: ranks edge against the chance of landing (raw EV favours long prices).
    George 6 Nov: 8+ reb $7.75 (+24%, 16%) and 20+ pts $3.70 (+14%, 31%) are near-equal here, so
    both are shown (best per stat), not one picked; he scored 33."""
    p, x = b["p"], b["price"]
    f = max((p * x - 1) / (x - 1), 0.0)
    return float(p * np.log1p(f * (x - 1)) + (1 - p) * np.log1p(-f)) if f > 0 else float(b["ev"]) * 1e-3


RARE_HR, RARE_W = 0.15, 0.5  # a line he hits under 15% of the time ranks at half weight (Traore 4+ reb 1/8, scored 19)


def _rare(b: dict) -> bool:
    return b.get("own_hr") is not None and b["own_hr"] < RARE_HR


def rank(b: dict) -> float:
    """Ranking score: Kelly growth, halved for lines he rarely reaches (lean into what he's good at)."""
    hr = b.get("own_hr")
    return growth(b) * (RARE_W if hr is not None and hr < RARE_HR else 1.0)


def _combo_ra(sim: dict) -> bool:
    """A guard/wing who does both: projected 3.5+ rebounds and 3.5+ assists."""
    return float(sim["reb"].mean()) >= 3.5 and float(sim["ast"].mean()) >= 3.5


LEG_FLOOR = 0.20  # each combo leg needs this chance tonight (a hard 35-50% weak-leg rule failed on this week's 11 combos)
RA_SHAPES = ((5, 5), (6, 5), (5, 6), (6, 6), (7, 5), (5, 7), (7, 7))
PR_SHAPES = ((8, 7), (10, 7), (12, 7), (15, 7), (10, 10), (12, 10), (15, 10), (18, 10), (12, 12))  # the user's P/R combos


def _big(sim: dict) -> bool:
    """A big for the P/R ladder: projected 6+ rebounds."""
    return float(sim["reb"].mean()) >= 6.0


def ra_ladder(a: dict, sim: dict, mk: dict, books: dict, c: float, pr: bool = False) -> list[dict]:
    """2-leg SGM ladder: R/A for a combo guard/wing, or P/R for a big (pr=True). Honest chance (sim, anchored,
    real-game link) and, per book, the builder estimate from its own legs. 'bet_at' clears MIN_EV."""
    out = []
    for x_, y_ in (PR_SHAPES if pr else RA_SHAPES):
        lg = [("pts", x_), ("reb", y_)] if pr else [("reb", x_), ("ast", y_)]
        if any(x not in mk for x in lg):
            continue
        pj, lab = anchored(a, sim, lg)
        legp = [anchored(a, sim, [x])[0] for x in lg]
        if min(legp) < LEG_FLOOR:
            continue  # a lottery leg: the combo is only as good as it
        strong = lg[int(np.argmax(legp))]
        rho = _p(sim, lg) / max(_p(sim, [lg[0]]) * _p(sim, [lg[1]]), 1e-9)
        rho = max(rho, ra_lift(a, lg) or rho)
        pk = mk[lg[0]] * mk[lg[1]] * rho
        ph = pk + c * (pj - pk)
        if ph <= 0:
            continue
        ests = {}
        for bk in SGM_BOOKS:
            own = [books.get(bk, {}).get(x) for x in lg]
            if all(own):
                ests[bk] = builder_factor(own, {s for s, _ in lg}) * float(np.prod(own))
        ests = {bk: v for bk, v in ests.items() if v <= SGM_RANGE[1]}  # past $15 the builder factors aren't calibrated
        best = max(ests.items(), key=lambda kv: kv[1]) if ests else (None, None)
        out.append({"bet": f"{x_}p/{y_}r" if pr else f"{x_}r/{y_}a", "p": ph, "fair": 1 / ph, "bet_at": (1 + MIN_EV) / ph,
                    "book": best[0], "est": best[1], "ev": best[1] * ph - 1 if best[1] else None, "hist": lab,
                    "strong": f"{strong[1]}+ {strong[0]}", "strong_p": max(legp),
                    "calibrated": True})
    return out


def bets_for(a: dict, sim: dict, mk: dict, books: dict, c: float, floor: float = 0.0) -> list[dict]:
    stats = {s.get("stat") for s in a["sits"] if s.get("stat") in ("ast", "reb")}
    focus = stats or {"pts", "reb", "ast"}
    out = []
    # singles (single-stat and combined rungs) at the user's books
    for bk in SGM_BOOKS:
        for (st, k), price in books.get(bk, {}).items():
            if "+" in st or not (SINGLE_RANGE[0] <= price <= SINGLE_RANGE[1]) or (st, k) not in mk:
                continue  # combined-stat rungs: build them as SGM legs instead; other stats are priced for the "also" list
            if st in ("reb", "ast") and k < SINGLE_MIN_LINE:
                continue  # user, 8 Nov: no 3+ reb / 3+ ast singles (Wendell Carter 3+ ast); SGM legs can still use 3s
            if not plausible(a, [(st, k)]):
                continue
            (pm, hist_lab), pk = anchored(a, sim, [(st, k)]), mk[(st, k)]
            ph = pk + c * (pm - pk)
            ev = price * ph - 1
            if ev >= floor:
                out.append({"kind": "single", "own_hr": hit_rate(a, [(st, k)])[0], "focus": st in focus, "lean": ev < MIN_EV, "bet": f"{k}+ {st}", "book": bk, "price": price, "p_model": pm, "p_mkt": pk, "hist": hist_lab,
                            "p": ph, "ev": ev, "take_at": 1.10 / ph, "outlier": price * pk / 1.0 > OUTLIER})
    # SGMs from one book's ladder
    for bk in SGM_BOOKS:
        bp = {key: v for key, v in books.get(bk, {}).items() if key[0] in LEG_MIN and key[1] >= LEG_MIN[key[0]]}
        legs = {s: [k for (st, k) in bp if st == s and _p(sim, [(s, k)]) >= LEG_MIN_P] for s in LEG_MIN}
        for n_legs in (2, 3):
            for ss in combinations(("pts", "reb", "ast"), n_legs):
                if not set(ss) & focus:
                    continue
                for ks in product(*(legs[s] for s in ss)):
                    lg = list(zip(ss, ks))
                    if not plausible(a, lg):
                        continue
                    pj_raw = _p(sim, lg)
                    pj, hist_lab = anchored(a, sim, lg)
                    med = books.get("au_median", {})
                    if any(x not in med for x in lg):
                        continue
                    own = [bp.get(x) for x in lg]
                    if all(own) and {s for s, _ in lg} in ({"reb", "ast"}, {"pts", "reb"}):  # R/A, P/R: own legs x shape factor
                        est = builder_factor(own, {s for s, _ in lg}) * float(np.prod(own))
                    else:
                        est = SGM_FACTOR * float(np.prod([med[x] for x in lg]))
                    if not (SGM_RANGE[0] <= est <= SGM_RANGE[1]) or any(x not in mk for x in lg) or pj <= 0:
                        continue
                    rho = pj_raw / float(np.prod([_p(sim, [x]) for x in lg]))  # model's within-player correlation
                    if len(lg) == 2:  # combo guard (R/A) or big (P/R): real games link the legs more than the sim when minutes swing
                        rho = max(rho, ra_lift(a, lg) or rho)
                    pk = float(np.prod([mk[x] for x in lg])) * rho
                    ph = pk + c * (pj - pk)
                    ev = est * ph - 1
                    if ev >= floor:  # below SGM_MIN_EV it's shown as the spot's SGM version, not a card bet
                        out.append({"kind": "SGM", "alt": ev < SGM_MIN_EV, "hist": hist_lab, "bet": "/".join(f"{k}{s[0]}" for s, k in lg), "book": bk, "price": est,
                                    "p_model": pj, "p_mkt": pk, "p": ph, "ev": ev, "take_at": 1.10 / ph, "outlier": False,
                                    "legs": (" × ".join(f"${x:.2f}" for x in own) if all(own) and {s for s, _ in lg} in ({"reb", "ast"}, {"pts", "reb"})
                                             else " × ".join(f"${med[x]:.2f}" for x in lg))})
    out.sort(key=lambda b: -rank(b))
    best, seen = [], set()
    for b in out:  # best per kind+shape
        shape = b["kind"] + "".join(ch for ch in b["bet"] if ch.isalpha() or ch == "+")
        if shape not in seen:
            seen.add(shape)
            best.append(b)
    for b in best:  # a line he rarely reaches is never a card tip, only filler (Sharpe 8+ ast: 2/9 this season, 0/62 last)
        if b["kind"] == "single" and b.get("own_hr") is not None and b["own_hr"] < RARE_HR:
            b["lean"] = True
    a["by_stat"] = [b for b in best if b["kind"] == "single"]  # best rung per stat, every stat (the "also" list)
    best = [b for b in best if b["kind"] == "SGM" or b.get("focus", True)]  # the spot's own bets: the stats its angle drives
    main = [b for b in best if not b.get("alt") and not b.get("lean")][:4]
    if not main:  # no bet clears the bar: keep the best lean for the card's filler
        main = [b for b in best if b.get("lean")][:1]
    sgm = next((b for b in best if b["kind"] == "SGM"), None)
    if sgm is not None and sgm not in main:
        main.append(sgm)
    return lead_with_single(main)


def apply_angle_h(a: dict, bets: list[dict], c: float, fac: dict, grp: str) -> list[str]:
    """Angle H (docs/opp_position.md): a role spot in a bad opposition spot for the bet's stat is graded down,
    a big's is passed (starter-out bigs -14% at Sportsbet, both seasons); a role spot or assist bounce in a good
    one gets +BOOST_C confidence. Opponent alone is below the margin, so price plays are untouched."""
    kinds = {x["kind"] for x in a["sits"]}
    bounce = {x.get("stat") for x in a["sits"] if x["kind"] == "bounce"}
    notes = []
    for b in bets:
        act, note = angle_h(fac, b, kinds, grp, bounce)
        if not act:
            continue
        b["h"] = act
        b["hist"] = (f"{b['hist']} · " if b.get("hist") else "") + note
        if note not in notes:
            notes.append(note)
        if act == "gate":
            b["lean" if b["kind"] == "single" else "alt"] = True
        elif act == "boost":
            b["c"] = min(c + BOOST_C, CONF_MAX)
            b["p"] = b["p_mkt"] + b["c"] * (b["p_model"] - b["p_mkt"])
            b["ev"] = b["price"] * b["p"] - 1
            b["take_at"] = 1.10 / b["p"]
            if b["kind"] == "single":
                b["lean"] = b["ev"] < MIN_EV or (b.get("own_hr") is not None and b["own_hr"] < RARE_HR)
            else:
                b["alt"] = b["ev"] < SGM_MIN_EV
    return notes


def lead_with_single(bets: list[dict]) -> list[dict]:
    """The card bet is the best firm single on the spot's own stats; combos stay listed as the SGM version.
    Live 4-9 Nov: combos 0/6 (exp 1.1, -6.0u) vs the same-stat single 1/6 (exp 2.0, -2.9u); the strong leg kept
    landing and the weaker leg kept missing (Amen 9r/4a with 9 reb, Achiuwa 10p/7r with 14 reb). Stable: order kept otherwise."""
    firm = lambda b: b["kind"] == "single" and not b.get("lean") and not b.get("outlier") and not b.get("alt")
    return sorted(bets, key=lambda b: not firm(b))


BETR_RUNGS = {"pts": (8, 10, 12, 15, 18, 20, 25), "reb": (3, 5, 7, 10, 12), "ast": (3, 5, 7, 10)}


def min_price(pm: float, c: float, ev: float = MIN_EV) -> float | None:
    """Lowest price at which the honest EV clears `ev`, before the market is known: the honest
    chance is the market's real chance at that price moved c of the way to the model."""
    for x in np.arange(1.30, 15.01, 0.05):
        r = market_ratio(x) / x
        if x * (r + c * (pm - r)) >= 1 + ev:
            return float(x)
    return None


def ladder_now(a: dict, sim: dict, c: float) -> list[dict]:
    """Rungs worth checking when his props post: model chance (anchored to matching games) and the
    price that would make it a bet. Only the stats the situation drives (all for a role change)."""
    stats = {s.get("stat") for s in a["sits"] if s.get("stat") in ("ast", "reb")} or set(BETR_RUNGS)
    out = []
    for st in ("pts", "reb", "ast"):
        if st not in stats:
            continue
        for k in BETR_RUNGS[st]:
            if not plausible(a, [(st, k)]):
                continue
            pm, lab = anchored(a, sim, [(st, k)])
            if 0.12 <= pm <= 0.75:
                mp = min_price(pm, c)
                if mp is not None and mp <= SINGLE_RANGE[1]:
                    out.append({"bet": f"{k}+ {st}", "p_model": pm, "fair": 1 / pm, "bet_at": mp, "hist": lab})
    return out


def grade(ev: float, c: float, kinds: set | None = None, outlier: bool = False) -> str:
    """A needs a lineup role / backup / emerging potentials spot; conversion bounces and minutes trends cap at B."""
    if outlier:
        return "C"
    strong = bool(kinds and "role" in kinds)
    if c >= 0.28 and ev >= 0.10 and strong:
        return "A"
    if ev >= 0.07:
        return "B"
    return "C"


STAKE = {"A": "1u", "B": "0.5u", "C": "0.25u"}


# ---------------------------------------------------------------- the slate
def build(day: date, extra_out: dict | None = None, include: set | None = None, news: dict | None = None) -> dict:
    """All spots for a day. extra_out: {team: {player_id}} to force absences (conditional cards).
    news: late lineup news the report doesn't have yet, {"out": [names], "start": [names], "bench": [names]}:
    out = absent tonight; start/bench = +/-NEWS_MIN minutes (the same both ways: Richard started for 16 min,
    Ziaire scored 17 off the bench)."""
    pg, trk = _frames(day)
    rep = _report(day, pg)
    idx = parse_player_index(get_player_index("2026-27"))
    team_now = dict(idx.select("player_id", "team_now").iter_rows())
    pid_name = dict(idx.select("player_id", "player_name").iter_rows())
    key_pid = {player_key(n): p for p, n in pid_name.items()}
    games = sorted(rep["matchup"].drop_nulls().unique().to_list())
    teams = {t: m for m in games for t in m.split("@")}
    gt = _game_table(pg, trk, team_now)
    gt_last = _game_table(pg, trk, team_now, str(int(SEASON) - 1))
    cur = pg.filter(pl.col("season_id") == SEASON)
    mpg = dict(cur.group_by("player_id").agg(pl.col("minutes").mean()).iter_rows())
    r36 = dict(cur.group_by("player_id").agg(pl.col("reb").sum() / pl.col("minutes").sum().clip(1) * 36).iter_rows())
    # stars: 20+ ppg in the most recent of the last two seasons where he played 20+ games (Tatum missed 2025-26)
    last_ppg = {}
    for sid in (str(int(SEASON) - 2), str(int(SEASON) - 1)):
        s = pg.filter(pl.col("season_id") == sid).group_by("player_id").agg(pl.col("pts").mean(), pl.len())
        last_ppg.update(dict(s.filter(pl.col("len") >= 20).select("player_id", "pts").iter_rows()))
    trk_last = trk.filter(pl.col("season_id") == str(int(SEASON) - 1))
    ca, cr = _conversions(trk)
    grp = dict(position_groups(positions()).iter_rows())
    from nba_ladders.matchup import multipliers

    opp_h = opp_tonight(day)  # angle H factors (opp_position.py, the version the market test used)
    from nba_ladders.usage import tonight as usage_tonight

    usage_fn = lambda p, out: usage_tonight(p, set(out), day)
    spreads = _spreads(day)
    odds = _odds(day)
    # games already tipped when the odds were taken: prices are in-play, so no bets (DEN@IND 7 Nov)
    started = set()
    if odds.height:
        snap = datetime.strptime(odds["snapshot"].max(), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        for ct, h, a_ in odds.select("commence_time", "home_team", "away_team").unique().iter_rows():
            if datetime.fromisoformat(ct) <= snap:
                started |= {TEAM_ABBR.get(h), TEAM_ABBR.get(a_)}
    filed = set(rep.filter(pl.col("status") != "Not submitted")["team"].drop_nulls()) - set(
        rep.filter(pl.col("status") == "Not submitted")["team"].drop_nulls())
    status = {}
    for r in rep.filter(pl.col("player_key").is_not_null()).iter_rows(named=True):
        pid = key_pid.get(r["player_key"])
        if pid is not None:
            status[pid] = (r["status"], _p_sit(r), r["reason"] or "")
    shift = {}
    for kind, names in (news or {}).items():
        for nm_ in names:
            pid = key_pid.get(player_key(nm_))
            if pid is None:
                print(f"news: no player matching {nm_!r}")
                continue
            if kind == "out":
                status[pid] = ("Out (news)", 1.0, "late lineup news")
            else:
                shift[pid] = NEWS_MIN if kind == "start" else -NEWS_MIN
    id_name = {**pid_name, **{("mpg", p): v for p, v in mpg.items()}, **{("grp", p): v for p, v in grp.items()}, **{("r36", p): v for p, v in r36.items()}}
    spots, leans, near, waiting, looked, evaluated = [], [], [], [], 0, []
    for team, m in teams.items():
        if team in started:
            continue
        opp = next(x for x in m.split("@") if x != team)
        roster = [p for p, t in team_now.items() if t == team and mpg.get(p, 0) >= ROT_MPG]
        out_t = {p for p in roster if status.get(p, ("", 0, ""))[1] >= OUT_P} | (extra_out or {}).get(team, set())
        unc = {p: status[p][1] for p in roster if p in status and SAFE_P <= status[p][1] < OUT_P and p not in out_t}
        wide = [p for p, t in team_now.items() if t == team and mpg.get(p, 0) >= LINEUP_MPG]
        t_in = {p for p in wide if p not in out_t and status.get(p, ("", 0, ""))[1] < SAFE_P}
        cands = cur.filter(pl.col("team") == team).group_by("player_id").agg(pl.col("minutes").tail(5).mean().alias("m5"))
        for pid, m5 in cands.iter_rows():
            if team_now.get(pid) != team or m5 < PRICE_MPG or pid in out_t:
                continue
            own_p = status.get(pid, ("", 0, ""))[1]
            if own_p >= SAFE_P:
                continue
            if last_ppg.get(pid, 0) >= STAR_PPG and not (include and pid_name.get(pid) in include):
                continue  # stars out (user, 8 Nov: Harden/Mobley on the card); `include` still prices them on request
            looked += 1
            g = gt.filter(pl.col("player_id") == pid, pl.col("team") == team)
            last = trk_last.filter(pl.col("player_id") == pid)
            gl = gt_last.filter(pl.col("player_id") == pid, pl.col("team") == team)
            gme = grp.get(pid, "W")
            opp_f = opp_h.get((opp, gme), {})  # opponent x position, every stat (opp_position.py)
            opp_tab = {o: f for (o, gg), f in opp_h.items() if gg == gme}
            a = analyse(pid, pid_name.get(pid, str(pid)), team, opp, g, last, out_t, unc, id_name,
                        ca.get(pid, 0.55), cr.get(pid, 0.55), opp_f, spreads.get(team), gl, opp_tab, gme, t_in, usage_fn)
            forced = include is not None and a is not None and a["name"] in include
            if a is not None and pid in shift:
                a["proj_min"] += shift[pid]
                verb = "starting" if shift[pid] > 0 else "coming off the bench"
                a["sits"].append({"kind": "role" if shift[pid] > 0 else "news", "news": True,
                                  "text": f"{verb} tonight (lineup news): {shift[pid]:+.1f} min"})
            if a is not None and (cut := own_status_cut(*status.get(pid, ("", 0, ""))[::2])):
                a["proj_min"] -= cut  # he plays through it, on fewer minutes (Allen 9 Nov: Q knee, 19 min vs 31)
            if a is not None:
                a["star"] = last_ppg.get(pid, 0) >= STAR_PPG  # stars are in, but a star with no angle ranks behind the second tier
            if a is None:
                continue
            if (m5 < CAND_MPG or not {x["kind"] for x in a["sits"]} & PRIMARY) and not forced:
                # a 3-game volume spike alone regressed (replay: -40%); it only supports a real spot.
                # These players go to the next-best pool only (projection vs price, no lineup angle).
                a["opp_line"] = opp_h_line(opp_h.get((opp, gme), {}), opp, gme, multipliers(opp, day)["pace"])
                a["opp_comps"] = opp_comps(cur, grp, opp, gme)
                a["unc"] = {pid_name.get(p, str(p)): v for p, v in unc.items()}
                em = {x.get("stat") for x in a["sits"] if x["kind"] == "emerging"}
                a["trend"] = "min" in em and bool(em & {"reb", "ast"})  # minutes AND chances up (Burries 8 Nov)
                a["sits"] = a["sits"] + [{"kind": "proj", "text": ("Trend: minutes and chances both up. " if a["trend"] else
                                                                   "No lineup angle: projection vs price. ") + a["opp_line"]}]
                c = max(CONF_MIN, confidence(a, sum(unc.values()), team in filed) - 0.10 * own_p - 0.05)
                a["own_p"] = own_p
                sim = simulate(a)
                a["combo_ra"], a["big"] = _combo_ra(sim), _big(sim)
                bs = [b for b in bets_for(a, sim, *market_probs(odds, a["name"]), c, floor=NEXT_FLOOR) if not b.get("outlier")]
                if bs:
                    a.update(conf=c, sim_means={k: float(v.mean()) for k, v in sim.items()},
                             min_p10_90=(float(np.percentile(sim["minutes"], 10)), float(np.percentile(sim["minutes"], 90))), bets=bs)
                    for b in bs:
                        b["grade"] = "C"
                    near.append(a)
                continue
            a["opp_line"] = opp_h_line(opp_h.get((opp, gme), {}), opp, gme, multipliers(opp, day)["pace"])
            a["opp_comps"] = opp_comps(cur, grp, opp, grp.get(pid, "W"))
            a["unc"] = {pid_name.get(p, str(p)): v for p, v in unc.items()}
            c = max(CONF_MIN, confidence(a, sum(unc.values()), team in filed) - 0.10 * own_p)
            a["own_p"] = own_p
            sim = simulate(a)
            a["combo_ra"], a["big"] = _combo_ra(sim), _big(sim)
            if forced:
                a["sim"] = sim
                a["mk_books"] = market_probs(odds, a["name"])
                a["c"] = c
                evaluated.append(a)
            mk, books = market_probs(odds, a["name"])
            bs = bets_for(a, sim, mk, books, c)
            notes_h = apply_angle_h(a, bs, c, opp_h.get((opp, grp.get(pid, "W")), {}), grp.get(pid, "W"))
            a.update(conf=c, sim_means={k: float(v.mean()) for k, v in sim.items()},
                     min_p10_90=(float(np.percentile(sim["minutes"], 10)), float(np.percentile(sim["minutes"], 90))), bets=bs)
            if a["combo_ra"]:
                a["ra_ladder"] = ra_ladder(a, sim, mk, books, c)
            if a["big"]:
                a["pr_ladder"] = ra_ladder(a, sim, mk, books, c, pr=True)
            kinds = {x["kind"] + ("-vol" if x["kind"] == "emerging" and x.get("stat") in ("ast", "reb") else "") for x in a["sits"]}
            est = (bool(a["key_new"]) and a["n_comp"] == 0 and "role" in kinds) or (any(x["kind"] == "role" for x in a["sits"]) and all(x.get("news") for x in a["sits"] if x["kind"] == "role"))
            for b in bs:
                b["grade"] = grade(b["ev"], b.get("c", c), kinds, b.get("outlier", False))
                if est and b["grade"] in "AB":
                    b["grade"] = "C"  # estimated role (no game without them yet): live 2/11, model expected 4.5
                if b.get("h") == "down":
                    b["grade"] = {"A": "B", "B": "C"}.get(b["grade"], "C")
            if notes_h:
                a["opp_line"] = (a.get("opp_line", "") + " · " + "; ".join(notes_h)).strip(" ·")
            if any(not b.get("alt") and not b.get("lean") for b in bs):
                spots.append(a)
            elif any(b.get("lean") for b in bs):
                leans.append(a)
            elif not mk:  # a real spot, but the books haven't posted his props yet
                a["ladder"] = ladder_now(a, sim, c)
                if a["ladder"]:
                    waiting.append(a)
    spots.sort(key=lambda a: -max(b["ev"] * a["conf"] for b in a["bets"]))
    leans.sort(key=lambda a: -max(b["ev"] for b in a["bets"]))
    return {"day": day, "news": news, "spots": spots, "leans": leans, "near": near, "waiting": waiting, "evaluated": evaluated, "report_time": str(rep["report_time"].max()), "filed": filed, "teams": set(teams),
            "snapshot": odds["snapshot"].max() if odds.height else None, "looked": looked, "status": status,
            "pid_name": pid_name, "team_now": team_now, "mpg": mpg}


def stat_of(b: dict) -> str:
    return b["bet"] if b["kind"] == "SGM" else b["bet"].split("+ ")[1]


def _one_per_player(bets: list, per_player: int = 2) -> list:
    """Each player's best bet per stat (up to per_player), by Kelly growth, so one player's long
    rungs don't crowd out the others and his points don't hide behind his rebounds."""
    out, seen, count = [], set(), {}
    for a, b in sorted(bets, key=lambda ab: -growth(ab[1])):
        key = (a["name"], stat_of(b))
        if key in seen or count.get(a["name"], 0) >= per_player:
            continue
        seen.add(key)
        count[a["name"]] = count.get(a["name"], 0) + 1
        out.append((a, b))
    return out


def conditional(day: date, base: dict) -> list[dict]:
    """For each Questionable rotation player (P(sits) 25-70%): spots that only exist if he sits."""
    out = []
    have = {(a["name"], b["bet"]) for a in base["spots"] for b in a["bets"]}
    for pid, (st, p, reason) in base["status"].items():
        team = base["team_now"].get(pid)
        if team not in base["teams"] or not (0.15 <= p < OUT_P) or base["mpg"].get(pid, 0) < KEY_MPG:
            continue
        alt = build(day, extra_out={team: {pid}}, news=base.get("news"))
        new = [(a, b) for a in alt["spots"] if a["team"] == team for b in a["bets"]
               if (a["name"], b["bet"]) not in have and not b.get("alt") and not b.get("outlier") and b["ev"] >= MIN_EV]
        out.append({"player": base["pid_name"].get(pid, str(pid)), "team": team, "p": p, "reason": reason,
                    "mpg": base["mpg"].get(pid, 0.0), "bets": _one_per_player(new)[:4]})
    out.sort(key=lambda c: -c["mpg"])
    return out


# ---------------------------------------------------------------- render
def _fmt_bet(b):
    est = "est. " if b["kind"] == "SGM" else ""
    return (f"**{b['bet']}** ({b['book'].replace('_au', '')}) {est}${b['price']:.2f} · fair ${1 / b['p']:.2f} · "
            f"EV {b['ev']:+.0%} · take at **${b['take_at']:.2f}**")


def card_bets(res: dict, n: int = CARD_N, cond: list | None = None) -> list[tuple[dict, dict]]:
    """Top n tips: one per spot (grade, then honest EV); filled with leans, then the best
    'if X sits' bets (labelled), if fewer qualify."""
    ok = lambda b: not b.get("outlier") and not b.get("alt") and not b.get("lean")
    ranked = [(a, next(b for b in a["bets"] if ok(b))) for a in res["spots"] if any(ok(b) for b in a["bets"])]
    ranked = sorted(ranked, key=lambda ab: ("ABC".index(ab[1]["grade"]), -rank(ab[1])))
    for a in res.get("leans", []):
        b = next((x for x in a["bets"] if x.get("lean") and not x.get("outlier") and not _rare(x) and x.get("h") != "gate"), None)
        if b is not None:
            ranked.append((a, b))
    card, per_game, per_dog = [], {}, {}
    for a, b in ranked:  # one game's tips share one blowout risk: cap them
        if len(card) >= n:
            break
        game = frozenset((a["team"], a["opp"]))
        dog = a.get("spread") is not None and a["spread"] >= DOG_SPREAD
        if per_game.get(game, 0) >= PER_GAME or (dog and per_dog.get(a["team"], 0) >= DOG_CAP):
            continue
        per_game[game] = per_game.get(game, 0) + 1
        per_dog[a["team"]] = per_dog.get(a["team"], 0) + dog
        card.append((a, b))
    have = {a["name"] for a, _ in card}
    extra = sorted(((c, a, b) for c in (cond or []) for a, b in c["bets"]), key=lambda x: -growth(x[2]))
    for c, a, b in extra:
        if len(card) >= n:
            break
        if a["name"] not in have:
            card.append((a, {**b, "if_sits": f"{c['player']} sits ({c['p']:.0%})"}))
            have.add(a["name"])
    return card


def price_plays(res: dict, card: list, n: int = PRICE_N) -> list[tuple[dict, dict]]:
    """Top n projection-vs-price bets with no lineup angle (the 'near' pool): one per player, honest EV >= MIN_EV,
    not rare lines or price outliers, ranked by Kelly growth like the card; at most one per game and one per
    stat (0/10 on 10-11 Nov: three rebound overs a night, two from one game). 0.25u each."""
    have = {a["name"] for a, _ in card}
    pool = []
    for a in res.get("near", []):
        if a["name"] in have:
            continue
        ok = sorted((b for b in a["bets"] if not b.get("outlier") and b["ev"] >= MIN_EV and not _rare(b)), key=lambda b: -growth(b))
        if ok:  # his best firm single leads, as on the card; a combo only when no single clears
            pool.append((a, lead_with_single([{**b, "lean": False} for b in ok])[0]))
    out, games, stats = [], set(), set()
    for a, b in sorted(pool, key=lambda ab: -growth(ab[1])):
        game = frozenset((a.get("team"), a.get("opp"))) if a.get("team") else None
        stat = "combo" if b.get("kind") == "SGM" else b["bet"].split()[-1]
        if (game is not None and game in games) or stat in stats:
            continue
        games.add(game)
        stats.add(stat)
        out.append((a, b))
        if len(out) >= n:
            break
    return out


def next_best(res: dict, card: list, cond: list | None = None, reserves: int = 2) -> list[tuple[dict, dict]]:
    """When the card has fewer than MIN_TIPS: the best of the rest, one per player, by honest EV.
    Pool: spots and leans not on the card, conditional bets, and players with no lineup angle
    (projection vs price only). Returns the picks needed to reach MIN_TIPS plus a few reserves."""
    need = MIN_TIPS - sum(not b.get("if_sits") and not b.get("lean") for _, b in card)  # IF bets may not trigger and leans are filler: neither counts
    if need <= 0:
        return []
    have = {a["name"] for a, _ in card}
    pool = []
    for a in res["spots"] + res.get("leans", []) + res.get("near", []):
        for b in a["bets"]:
            if not b.get("outlier") and b["ev"] >= NEXT_FLOOR and not _rare(b):
                pool.append((a, b))
    for c in cond or []:
        for a, b in c["bets"]:
            pool.append((a, {**b, "if_sits": f"{c['player']} sits ({c['p']:.0%})"}))
    # a real angle beats a bare projection at the same EV
    angle = lambda a: any(x["kind"] in PRIMARY for x in a["sits"])
    weight = lambda a: 1.3 if angle(a) else 1.2 if a.get("trend") else STAR_NO_ANGLE if a.get("star") else 1.0
    pool.sort(key=lambda ab: -growth(ab[1]) * weight(ab[0]))
    out = []
    for a, b in pool:
        if a["name"] in have:
            continue
        have.add(a["name"])
        out.append((a, b))
        if len(out) >= need + reserves:
            break
    return out


def kelly(b: dict) -> float:
    """Full-Kelly stake as a share of bankroll (stake a fraction of it: the honest p is still an estimate)."""
    return max((b["p"] * b["price"] - 1) / (b["price"] - 1), 0.0)


def also_bets(a: dict, b: dict) -> list[dict]:
    """His other +EV singles, best rung per other stat, by Kelly growth (Traore 4+ reb: also his pts / ast)."""
    pool = a.get("by_stat") or [x for x in a["bets"] if x["kind"] == "single"]
    legs = set(stat_of(b).split("/")) if b["kind"] != "SGM" else set()
    return sorted((x for x in pool if not x.get("outlier") and x["ev"] >= 0 and stat_of(x) not in legs | {stat_of(b)}),
                  key=lambda x: -growth(x))


def also_bet(a: dict, b: dict) -> dict | None:
    xs = also_bets(a, b)
    return xs[0] if xs else None


def _also_md(a: dict, b: dict) -> str:
    ra = sorted((r for r in (a.get("ra_ladder") or []) + (a.get("pr_ladder") or []) if r["ev"] is not None and r["ev"] >= MIN_EV),
                key=lambda r: -r["ev"])[:3]
    return "".join(f" · SGM **{r['bet']}** est ${r['est']:.2f} (fair ${r['fair']:.2f}, bet at ${r['bet_at']:.2f})" for r in ra) + "".join(f" · also **{x['bet']}** ${x['price']:.2f} ({x['p']:.0%}, EV {x['ev']:+.0%}{', ≈ fair' if x['ev'] < MIN_EV else ''}, "
                   f"Kelly {kelly(x):.1%})" for x in also_bets(a, b))


def proj_line(a: dict, b: dict) -> str:
    """'Exp 18 min (9-28) · 5.5p / 3.1r / 1.0a · 5+ reb: 17% (fair $5.94)'."""
    s = a.get("sim_means")
    if not s:
        return ""
    lo, hi = a.get("min_p10_90", (None, None))
    rng = f" ({lo:.0f}–{hi:.0f})" if lo is not None else ""
    return (f"Exp {s['minutes']:.0f} min{rng} · {s['pts']:.1f}p / {s['reb']:.1f}r / {s['ast']:.1f}a · "
            f"{b['bet']}: {b['p']:.0%} chance (fair ${1 / b['p']:.2f})")


def _why(a: dict, b: dict) -> str:
    """The situation that drives this bet's stat (role spots drive every stat)."""
    leg = {"p": "pts", "r": "reb", "a": "ast"}
    stats = ({leg[x[-1]] for x in b["bet"].split("/")} if b["kind"] == "SGM"
             else set(b["bet"].split("+ ")[1].split("+")))
    for x in a["sits"]:
        if x["kind"] == "role" or x.get("stat") in stats:
            return x["text"]
    return a["sits"][0]["text"]


def render(res: dict, cond: list[dict]) -> str:
    day = res["day"]
    unfiled = sorted(res["teams"] - res["filed"])
    L = [f"# Spots: {day} (US Eastern)", "",
         (f"Injury report {res['report_time']} ET · odds snapshot {res['snapshot']} · looked at {res['looked']} rotation "
          f"players, {len(res['spots'])} spots."),
         "" if not unfiled else f"\n**Not yet filed:** {', '.join(unfiled)}. Those spots are graded down; re-run after the 5 PM ET report.",
         "", ("**How to read this**: a spot needs a reason the player is different tonight (role change from tonight's "
              "lineup, emerging role, unlucky conversion on normal volume, backup big). The projection comes from his "
              "games that match tonight. **Fair and EV are honest**: they start from the market's own probability and "
              "move toward the model only by the confidence shown, so a +15% here means more than +80% did before. "
              "SGM prices are builder estimates (0.9× the legs on that book): take only at the 'take at' price or better.")]
    card = card_bets(res, cond=cond)
    L += ["", "## The card", "", "| Grade | Stake | Player | Bet | Projection | SGM version | Why |", "|---|---|---|---|---|---|---|"]
    for a, b in card:
        sg = next((x for x in a["bets"] if x["kind"] == "SGM" and x is not b), None)
        if b.get("if_sits"):
            L.append(f"| **If** | 0.5u | **{a['name']}** ({a['team']} vs {a['opp']}) | **only if {b['if_sits']}**: {_fmt_bet(b)} | {proj_line(a, b)}{_also_md(a, b)} | — | "
                     f"{_why(a, b)} |")
            continue
        L.append(f"| **{'Lean' if b.get('lean') else b['grade']}** | {'0.25u' if b.get('lean') else STAKE[b['grade']]} | **{a['name']}** ({a['team']} vs {a['opp']}) | {_fmt_bet(b)} | {proj_line(a, b)}{_also_md(a, b)} | "
                 f"{_fmt_bet(sg) if sg else '—'} | {_why(a, b)}" + (f"; {a['opp_line']}" if a.get('opp_line') else "") + " |")
    pp = price_plays(res, card)
    if pp:
        L += ["", "## Price plays (no lineup angle: projection vs price, 0.25u)", "",
              "| Player | Bet | Projection | Why |", "|---|---|---|---|"]
        for a, b in pp:
            L.append(f"| **{a['name']}** ({a['team']} vs {a['opp']}) | {_fmt_bet(b)} | {proj_line(a, b)} | {_why(a, b)} |")
    passed = [(a, b) for a in res["spots"] + res.get("leans", []) for b in a["bets"][:1] if b.get("h") == "gate"]
    if passed:
        L += ["", "## Passed: bad opposition spot (angle H, docs/opp_position.md)", ""]
        for a, b in passed:
            L.append(f"- **{a['name']}** ({a['team']} vs {a['opp']}): {_fmt_bet(b)}. {b['hist'].split(' · ')[-1]}")
    if res.get("waiting"):
        L += ["", "## Spots waiting for prices (props not posted yet)", ""]
        for a in sorted(res["waiting"], key=lambda a: -a["conf"]):
            s = a["sim_means"]
            L.append(f"- **{a['name']}** ({a['team']} vs {a['opp']}): {'; '.join(x['text'] for x in a['sits'])}. "
                     f"Exp {s['minutes']:.0f} min · {s['pts']:.1f}p / {s['reb']:.1f}r / {s['ast']:.1f}a. Bet at: "
                     + ", ".join(f"{r['bet']} {r['p_model']:.0%} → ${r['bet_at']:.2f}+" for r in a["ladder"]))
    shop = [(a, b) for a in res["spots"] for b in a["bets"] if b.get("outlier")]
    if shop:
        L += ["", "## Prices far longer than other books (check they're live before betting)", ""]
        for a, b in sorted(shop, key=lambda ab: -ab[1]["ev"])[:6]:
            L.append(f"- {a['name']} ({a['team']}): {_fmt_bet(b)} · market fair ${1 / b['p_mkt']:.2f}")
    if cond:
        L += ["", "## If they sit (conditional bets)", ""]
        for cnd in cond:
            L.append(f"- **If {cnd['player']} ({cnd['team']}) sits** (now P(sits) {cnd['p']:.0%}: {cnd['reason']}):")
            for a, b in cnd["bets"]:
                L.append(f"    - {a['name']}: {_fmt_bet(b)}. {_why(a, b)}")
    L += ["", "## Spots", ""]
    for a in res["spots"]:
        s = a["sim_means"]
        L += [f"### {a['name']} ({a['team']} vs {a['opp']}) · confidence {a['conf']:.2f}", ""]
        for x in a["sits"]:
            L.append(f"- **{x['kind']}**: {x['text']}")
        if a.get("opp_line"):
            L.append(f"- **matchup**: {a['opp_line']}")
        if a.get("own_p", 0) >= 0.2:
            L.append(f"- **his own status**: P(sits) {a['own_p']:.0%}. Props are void if he doesn't play; confidence reduced")
        if a["unc"]:
            L.append("- **uncertain teammates**: " + ", ".join(f"{n} P(sits) {p:.0%}" for n, p in a["unc"].items()))
        L.append(f"- **projection** ({a['why_sample'] or 'baseline'}): {s['minutes']:.0f} min ({a['min_p10_90'][0]:.0f}–"
                 f"{a['min_p10_90'][1]:.0f}), {s['pts']:.1f} pts / {s['reb']:.1f} reb / {s['ast']:.1f} ast · "
                 f"{a['pot36']:.1f} potential ast and {a['rc36']:.1f} reb chances per 36 at {a['ca']:.0%} / {a['cr']:.0%} career conversion"
                 + (f" · spread {a['spread']:+.1f}" if a["spread"] is not None else ""))
        def games_line(df, title, L=L):
            rows = df.select("game_date", "opponent", "minutes", "pts", "reb", "ast", "potential_ast", "reb_chances").rows()
            extra = df["also_out"].to_list() if "also_out" in df.columns else [""] * df.height
            L.append(f"- **{title}**: " + "; ".join(
                f"{d:%d %b} vs {o}: {mi} min {p}/{r}/{x}" + (f" ({pa} pot, {'–' if rc is None else rc} ch)" if pa is not None else "")
                + (f" [also out: {e}, counts less]" if e else "")
                for (d, o, mi, p, r, x, pa, rc), e in list(zip(rows, extra))[::-1]))

        if a.get("last5") is not None and a["last5"].height:
            games_line(a["last5"], "last 5 games (any lineup)")
        if a["sample"] is not None and a["sample"].height:
            this = a["sample"].filter(pl.col("season_id") == SEASON)
            n_last = a["sample"].height - this.height
            if this.height:
                games_line(this.tail(6), "games that match tonight, this season"
                           + (f" (+{n_last} last-season games in the projection)" if n_last else ""))
            elif n_last:
                L.append(f"- **games that match tonight**: none this season; projection uses {n_last} last-season games")
        L += ["", "| Bet | Book | Price | Model | Market | **Honest fair** | EV | Take at | Grade |", "|---|---|---|---|---|---|---|---|---|"]
        for b in a["bets"]:
            L.append(f"| {b['kind']} **{b['bet']}** | {b['book'].replace('_au', '')} | {'est. ' if b['kind'] == 'SGM' else ''}${b['price']:.2f}"
                     f"{' (' + b['legs'] + ')' if b.get('legs') else ''} | ${1 / b['p_model']:.2f} | ${1 / b['p_mkt']:.2f} | "
                     f"**${1 / b['p']:.2f}** | {b['ev']:+.0%} | ${b['take_at']:.2f} | {b['grade']}"
                     + (" · ⚠️ price far above the market: check it's live" if b.get("outlier") else "") + " |")
        L.append("")
    return "\n".join(L)


def save(res: dict, cond: list[dict]) -> None:
    rows = []
    shown = {(a["name"], b["bet"]) for a, b in card_bets(res, cond=cond) if not b.get("if_sits")}  # the tips actually on the card
    for a in res["spots"]:
        for i, b in enumerate(a["bets"]):
            rows.append({"day": res["day"], "player": a["name"], "team": a["team"], "kind": b["kind"], "bet": b["bet"],
                         "book": b["book"], "price": b["price"], "p": b["p"], "p_model": b["p_model"], "p_mkt": b["p_mkt"],
                         "ev": b["ev"], "grade": b["grade"], "conf": a["conf"], "card": (a["name"], b["bet"]) in shown, "alt": bool(b.get("alt")),
                         "situations": "; ".join(x["kind"] for x in a["sits"])})
    for c in cond:
        for a, b in c["bets"]:
            rows.append({"day": res["day"], "player": a["name"], "team": a["team"], "kind": b["kind"], "bet": b["bet"],
                         "book": b["book"], "price": b["price"], "p": b["p"], "p_model": b["p_model"], "p_mkt": b["p_mkt"],
                         "ev": b["ev"], "grade": "IF", "conf": a["conf"], "card": False, "alt": False,
                         "situations": "if " + c["player"] + " sits", "if_sits": c["player"]})
    for a, b in price_plays(res, card_bets(res, cond=cond)):
        rows.append({"day": res["day"], "player": a["name"], "team": a["team"], "kind": b["kind"], "bet": b["bet"],
                     "book": b["book"], "price": b["price"], "p": b["p"], "p_model": b["p_model"], "p_mkt": b["p_mkt"],
                     "ev": b["ev"], "grade": "PP", "conf": a["conf"], "card": False, "alt": False, "situations": "price"})
    for a, b in next_best(res, card_bets(res, cond=cond), cond):  # graded too, so the trend category gets a live record
        if b.get("if_sits"):
            continue
        rows.append({"day": res["day"], "player": a["name"], "team": a["team"], "kind": b["kind"], "bet": b["bet"],
                     "book": b["book"], "price": b["price"], "p": b["p"], "p_model": b["p_model"], "p_mkt": b["p_mkt"],
                     "ev": b["ev"], "grade": "NB", "conf": a["conf"], "card": False, "alt": False,
                     "situations": "trend" if a.get("trend") else "; ".join(x["kind"] for x in a["sits"])})
    out = config.DATA_PROCESSED / "spots" / f"{res['day']}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out)


def grade_day(day: date) -> pl.DataFrame:
    """Score a saved card: hits vs honest / model / market expectation, flat-stake P&L."""
    df = pl.read_parquet(config.DATA_PROCESSED / "spots" / f"{day}.parquet")
    if df.is_empty() or "player" not in df.columns:
        return pl.DataFrame()
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(pl.col("game_date") == day)
    box = {player_key(r["player_name"]): r for r in pg.iter_rows(named=True)}
    leg = {"p": "pts", "r": "reb", "a": "ast"}
    hits = []
    for r in df.iter_rows(named=True):
        b = box.get(player_key(r["player"]))
        trig = r.get("if_sits")
        if b is None or (trig and player_key(trig) in box):  # DNP, or the conditional never triggered (void)
            hits.append(None)
            continue
        if r["kind"] == "SGM":
            legs = [(leg[x[-1]], int(x[:-1])) for x in r["bet"].split("/")]
        else:
            k, st = r["bet"].split("+ ")
            legs = [(st, int(k))]
        hits.append(all(sum(b[s] for s in st.split("+")) >= k for st, k in legs))
    return df.with_columns(pl.Series("hit", hits, dtype=pl.Boolean))


LIVE_FROM = date(2026, 11, 4)  # first slate published before tip-off (earlier days are in-sample replays)


def record(upto: date, live: bool = True) -> dict | None:
    """Running record of graded slates before `upto`: card bets + triggered conditional bets, 1u flat."""
    rows = []
    for f in sorted((config.DATA_PROCESSED / "spots").glob("*.parquet")):
        d = date.fromisoformat(f.stem)
        if d >= upto or (live and d < LIVE_FROM) or (not live and d >= LIVE_FROM):
            continue
        pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(pl.col("game_date") == d)
        if pg.is_empty():
            continue
        g = grade_day(d)
        if g.is_empty():
            continue
        if "if_sits" not in g.columns:
            g = g.with_columns(pl.lit(None, dtype=pl.String).alias("if_sits"))
        g = g.filter(pl.col("hit").is_not_null(), pl.col("card") | pl.col("if_sits").is_not_null())
        rows.append(g.unique(["day", "player", "bet"], keep="first").select("day", "price", "p", "hit"))  # one bet two triggers (Poeltl: Barnes or Quickley) counts once
    if not rows:
        return None
    a = pl.concat(rows)
    return {"n": a.height, "hits": int(a["hit"].sum()), "exp": float(a["p"].sum()),
            "pnl": float((a["hit"].cast(int) * a["price"]).sum() - a.height), "days": a["day"].n_unique()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--no-conditional", action="store_true")
    ap.add_argument("--out", nargs="*", default=[], help="late scratches the report doesn't have yet")
    ap.add_argument("--start", nargs="*", default=[], help="starting tonight (news): +NEWS_MIN minutes")
    ap.add_argument("--bench", nargs="*", default=[], help="coming off the bench (news): -NEWS_MIN minutes")
    args = ap.parse_args()
    if args.grade:
        g = grade_day(args.date)
        played = g.filter(pl.col("hit").is_not_null())
        with pl.Config(tbl_rows=60, tbl_width_chars=220, fmt_str_lengths=40, tbl_hide_column_data_types=True):
            print(played.select("player", "kind", "bet", "price", "p", "p_model", "p_mkt", "ev", "grade", "card", "hit"))
        for lab, d in (("all bets", played), ("card (1 per spot)", played.filter(pl.col("card")))):
            if d.height:
                pnl = float((d["hit"].cast(int) * d["price"]).sum() - d.height)
                print(f"{lab}: {d['hit'].sum()}/{d.height} hit | honest expected {d['p'].sum():.1f} | model {d['p_model'].sum():.1f} | "
                      f"market {d['p_mkt'].sum():.1f} | flat 1u {pnl:+.2f}u")
        return
    news = {k: v for k, v in (("out", args.out), ("start", args.start), ("bench", args.bench)) if v}
    res = build(args.date, news=news or None)
    res["record"], res["replay"] = record(args.date), record(args.date, live=False)
    cond = [] if args.no_conditional else conditional(args.date, res)
    md = render(res, cond)
    out = config.ROOT / "docs" / "slates" / f"{args.date.isoformat()}-spots.md"
    out.write_text(md)
    from nba_ladders.spots_html import render_html

    out.with_suffix(".html").write_text(render_html(res, cond))
    card = card_bets(res, cond=cond)
    nb = next_best(res, card, cond)
    if nb:
        from nba_ladders.spots_html import render_next_html

        nxt = out.with_name(f"{args.date.isoformat()}-next-best.html")
        nxt.write_text(render_next_html(res, card, nb))
        print(f"card has {len(card)} tips: next best -> {nxt.name}: " + "; ".join(f"{a['name']} {b['bet']} ${b['price']:.2f} EV {b['ev']:+.0%}" for a, b in nb))
    save(res, cond)
    print(md[:3000])
    print(f"wrote {out} and .html")


if __name__ == "__main__":
    main()
