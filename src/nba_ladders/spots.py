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
from datetime import date, datetime, timedelta
from itertools import combinations, product

import numpy as np
import polars as pl

from nba_ladders import config
from nba_ladders.absences import position_groups
from nba_ladders.luck import K_CONV, league_conversion
from nba_ladders.luck import load as load_tracking
from nba_ladders.market import TEAM_ABBR, latest
from nba_ladders.scenarios import positions
from nba_ladders.screens import parse_player_index
from nba_ladders.sources.injury_report import ET, player_key, report_for_date
from nba_ladders.sources.nba_stats import get_player_index
from nba_ladders.sources.odds_api import load_cached

SEASON = "22026"
ROT_MPG = 20.0  # teammates whose absence changes a lineup
KEY_MPG = 24.0  # a "key" absence for role spots
CAND_MPG = 15.0  # players considered (recent minutes)
STAR_PPG = 20.0  # last season: stars are left out (sharply priced)
OUT_P = 0.70  # P(sits) at which a teammate counts as out tonight
SAFE_P = 0.60  # the player himself: P(sits) below this (props are void if he doesn't play; uncertainty costs confidence)
# Market calibration: actual hit rate / implied probability by price, 664k Sportsbet ladder rungs 2024-26
# (alt overs are priced short, more so at longer odds). Used to turn a price into the market's real probability.
MKT_PRICE = np.array([1.24, 1.74, 2.24, 2.76, 3.44, 4.5, 5.62, 7.34, 9.12, 11.79])
MKT_RATIO = np.array([0.934, 0.929, 0.899, 0.889, 0.856, 0.82, 0.789, 0.792, 0.769, 0.674])


def market_ratio(price: float) -> float:
    return float(np.interp(np.log(price), np.log(MKT_PRICE), MKT_RATIO))
CONF_MIN, CONF_MAX = 0.08, 0.35
SGM_FACTOR = 0.90  # builder ≈ this x product of the book's legs (user's slips: 0.87-1.2x)
LEG_MIN_P = 0.50  # each SGM leg live on its own
SGM_MIN_EV = 0.08
CARD_N = 10
LEG_MIN = {"pts": 8, "reb": 3, "ast": 3}
SGM_BOOKS = ("betr_au", "sportsbet")
SINGLE_RANGE = (2.5, 10.0)
SGM_RANGE = (3.5, 15.0)
FREED_CAP = 6.0  # minutes: the rough share-of-freed-minutes estimate when there's no game without them
PRIMARY = {"role", "bounce"}  # spot types that stand on their own
ROLE_MIN, ROLE_VOL = 2.0, 1.15  # a role spot needs +2 min or +15% volume vs his normal-lineup games
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
    rot = (cur.group_by("player_id", "team").agg(pl.col("minutes").mean().alias("mpg"), pl.len().alias("gp"))
           .filter(pl.col("mpg") >= ROT_MPG, pl.col("gp") >= 2))
    games = cur.select("team", "game_id").unique()
    played = cur.filter(pl.col("minutes") > 0).select("team", "game_id", "player_id")
    absent = (games.join(rot.select("team", "player_id"), on="team")
              .join(played, on=["team", "game_id", "player_id"], how="anti")
              .group_by("team", "game_id").agg(pl.col("player_id").alias("absent")))
    t = trk.filter(pl.col("season_id") == season).select("player_id", "game_id", "potential_ast", "reb_chances")
    return (cur.join(t, on=["player_id", "game_id"], how="left")
            .join(absent, on=["team", "game_id"], how="left")
            .with_columns(pl.col("absent").fill_null([]))
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


def _per36(df: pl.DataFrame, col: str) -> float | None:
    d = df.drop_nulls(col)
    return float(d[col].sum() / d["minutes"].sum() * 36) if d.height and d["minutes"].sum() > 0 else None


def _wmean(df: pl.DataFrame, col: str, w: np.ndarray) -> float:
    return float(np.sum(df[col].to_numpy() * w) / np.sum(w))


def analyse(pid: int, name: str, team: str, opp: str, g: pl.DataFrame, last: pl.DataFrame, tonight_out: set,
            tonight_unc: dict, id_name: dict, ca: float, cr: float, opp_f: dict, spread: float | None,
            g_last: pl.DataFrame | None = None) -> dict | None:
    """Situations, comparable games and the projection for one player. g = his games this season (current team)."""
    if g.height < 2:
        return None
    absent_sets = [set(a) for a in g["absent"].to_list()]
    played_share = {m: sum(m not in a for a in absent_sets) / len(absent_sets) for m in tonight_out}
    key_new = {m for m in tonight_out if played_share[m] >= 0.5}
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
        why = "games with " + ", ".join(sorted(id_name.get(m, str(m)) for m in key_new)) + " out" + (
            f" ({comp.height} this season, {last_comp.height} last season)" if last_comp is not None else "")
    elif key_new:
        sample, why = None, ""
    else:
        sample, why = comp.tail(6), "recent games with tonight's lineup"
    proj_min, pot36, rc36, pts36, n_comp = base_min, b_pot, b_rc, b_pts, 0
    if sample is not None and sample.height:
        w = _weights(sample.height)
        if last_comp is not None:  # last season counts half
            w = w * np.where(sample["season_id"].to_numpy() == SEASON, 1.0, 0.5)
        n_comp = int(comp.height + (0.5 * last_comp.height if last_comp is not None else 0))
        k = 3.0  # pseudo-games of baseline
        s_min = _wmean(sample, "minutes", w)
        proj_min = (s_min * w.sum() + base_min * k) / (w.sum() + k)
        def blend36(col, prior):
            d = sample.drop_nulls(col)
            if not d.height:
                return prior
            ww = _weights(d.height) * np.where(d["season_id"].to_numpy() == SEASON, 1.0, 0.5)
            v = float(np.sum(d[col].to_numpy() * ww) / np.sum(d["minutes"].to_numpy() * ww) * 36)
            return v if prior is None else (v * ww.sum() + prior * k) / (ww.sum() + k)
        pot36, rc36, pts36 = blend36("potential_ast", b_pot), blend36("reb_chances", b_rc), blend36("pts", b_pts)
    elif key_new:  # no game yet without them: a share of the freed minutes, mostly to the same position
        mine = id_name.get(("grp", pid), "W")
        freed = sum(id_name.get(("mpg", m), 30.0) * (0.25 if id_name.get(("grp", m), "W") == mine else 0.06) for m in key_new)
        proj_min = min(base_min + min(freed, FREED_CAP), 40.0)
        why = "no game yet with this lineup (estimated)"
    # ---- situations
    vol_up = max((pot36 or 0) / b_pot if b_pot else 1, (rc36 or 0) / b_rc if b_rc else 1)
    if key_new and (proj_min - base_min >= ROLE_MIN or (vol_up >= ROLE_VOL and n_comp >= 3)):
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
    rate = {"ast": pot36 * ca / 36 * opp_f.get("ast", 1.0), "reb": rc36 * cr / 36 * opp_f.get("reb", 1.0),
            "pts": pts36 / 36 * opp_f.get("pts", 1.0)}
    src = sample if sample is not None and sample.height >= 3 else g.tail(8)
    min_sd = max(4.0, float(src["minutes"].std() or 0.0))
    hist = pl.concat([g.select("season_id", "pts", "reb", "ast"), last.select("season_id", "pts", "reb", "ast")]
                     if last.height else [g.select("season_id", "pts", "reb", "ast")], how="vertical_relaxed")
    return {"pid": pid, "name": name, "team": team, "opp": opp, "sits": sits, "hist": hist, "key_new": key_new, "n_comp": n_comp,
            "why_sample": why, "sample": sample, "base_min": base_min, "proj_min": proj_min, "min_sd": min_sd,
            "rate": rate, "pot36": pot36, "rc36": rc36, "ca": ca, "cr": cr, "spread": spread, "opp_f": opp_f,
            "games": g.height, "last_games": last.height}


def simulate(a: dict, n: int = N_SIM) -> dict:
    """Minutes (with the spread's blowout curve) then Poisson counts at the projected rates."""
    from nba_ladders.blowout import CURVE, MID, SD_MARGIN, TYPICAL_SPREAD, role
    from nba_ladders.sim import draw_points

    m = RNG.normal(a["proj_min"], a["min_sd"], n)
    if a["spread"] is not None:
        curve = CURVE[role(a["proj_min"])]
        base = np.interp(np.abs(RNG.normal(TYPICAL_SPREAD, SD_MARGIN, 20_000)), MID, curve).mean()
        m = m + np.interp(np.abs(RNG.normal(abs(a["spread"]), SD_MARGIN, n)), MID, curve) - base
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
    return float(np.clip(c, CONF_MIN, cap))


# ---------------------------------------------------------------- pricing and bets
AU = ("sportsbet", "tab", "dabble_au", "betr_au", "neds", "ladbrokes_au", "pointsbetau")


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
            hit = pl.lit(True)
            for st, k in legs:
                hit = hit & (pl.sum_horizontal([pl.col(x) for x in st.split("+")]) >= k)
            n_hit = int(s.select(hit.alias("h"))["h"].sum())
            return (1 - HIST_W) * pm + HIST_W * n_hit / s.height, f"hit {n_hit}/{s.height} in games like tonight · {lab}"
        return pm, f"{lab} (old role)" if lab else ""
    return (pm if hr is None else (1 - HIST_W) * pm + HIST_W * hr), lab


def plausible(a: dict, legs) -> bool:
    """'He's never got 4': skip lines he almost never reaches, unless his minutes jump."""
    if a["proj_min"] - a["base_min"] >= BIG_ROLE_MIN:
        return True
    hr, _ = hit_rate(a, legs)
    return hr is None or hr >= PLAUSIBLE


def bets_for(a: dict, sim: dict, mk: dict, books: dict, c: float) -> list[dict]:
    stats = {s.get("stat") for s in a["sits"] if s.get("stat") in ("ast", "reb")}
    focus = stats or {"pts", "reb", "ast"}
    out = []
    # singles (single-stat and combined rungs) at the user's books
    for bk in SGM_BOOKS:
        for (st, k), price in books.get(bk, {}).items():
            if "+" in st or not (SINGLE_RANGE[0] <= price <= SINGLE_RANGE[1]) or st not in focus or (st, k) not in mk:
                continue  # combined-stat rungs: build them as SGM legs instead
            if not plausible(a, [(st, k)]):
                continue
            (pm, hist_lab), pk = anchored(a, sim, [(st, k)]), mk[(st, k)]
            ph = pk + c * (pm - pk)
            ev = price * ph - 1
            if ev >= MIN_EV:
                out.append({"kind": "single", "bet": f"{k}+ {st}", "book": bk, "price": price, "p_model": pm, "p_mkt": pk, "hist": hist_lab,
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
                    est = SGM_FACTOR * float(np.prod([med[x] for x in lg]))  # builders price off the market, not one book
                    if not (SGM_RANGE[0] <= est <= SGM_RANGE[1]) or any(x not in mk for x in lg) or pj <= 0:
                        continue
                    rho = pj_raw / float(np.prod([_p(sim, [x]) for x in lg]))  # model's within-player correlation
                    pk = float(np.prod([mk[x] for x in lg])) * rho
                    ph = pk + c * (pj - pk)
                    ev = est * ph - 1
                    if ev >= 0.0:  # below SGM_MIN_EV it's shown as the spot's SGM version, not a card bet
                        out.append({"kind": "SGM", "alt": ev < SGM_MIN_EV, "hist": hist_lab, "bet": "/".join(f"{k}{s[0]}" for s, k in lg), "book": bk, "price": est,
                                    "p_model": pj, "p_mkt": pk, "p": ph, "ev": ev, "take_at": 1.10 / ph, "outlier": False,
                                    "legs": " × ".join(f"${med[x]:.2f}" for x in lg)})
    out.sort(key=lambda b: -b["ev"])
    best, seen = [], set()
    for b in out:  # best per kind+shape
        shape = b["kind"] + "".join(ch for ch in b["bet"] if ch.isalpha() or ch == "+")
        if shape not in seen:
            seen.add(shape)
            best.append(b)
    main = [b for b in best if not b.get("alt")][:4]
    sgm = next((b for b in best if b["kind"] == "SGM"), None)
    if sgm is not None and sgm not in main:
        main.append(sgm)
    return main


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
def build(day: date, extra_out: dict | None = None) -> dict:
    """All spots for a day. extra_out: {team: {player_id}} to force absences (conditional cards)."""
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
    # stars: 20+ ppg in the most recent of the last two seasons where he played 20+ games (Tatum missed 2025-26)
    last_ppg = {}
    for sid in (str(int(SEASON) - 2), str(int(SEASON) - 1)):
        s = pg.filter(pl.col("season_id") == sid).group_by("player_id").agg(pl.col("pts").mean(), pl.len())
        last_ppg.update(dict(s.filter(pl.col("len") >= 20).select("player_id", "pts").iter_rows()))
    trk_last = trk.filter(pl.col("season_id") == str(int(SEASON) - 1))
    ca, cr = _conversions(trk)
    grp = dict(position_groups(positions()).iter_rows())
    from nba_ladders.matchup import group_factors, weakness_line

    gf = group_factors(day)
    spreads = _spreads(day)
    odds = _odds(day)
    filed = set(rep.filter(pl.col("status") != "Not submitted")["team"].drop_nulls()) - set(
        rep.filter(pl.col("status") == "Not submitted")["team"].drop_nulls())
    status = {}
    for r in rep.filter(pl.col("player_key").is_not_null()).iter_rows(named=True):
        pid = key_pid.get(r["player_key"])
        if pid is not None:
            status[pid] = (r["status"], _p_sit(r), r["reason"] or "")
    id_name = {**pid_name, **{("mpg", p): v for p, v in mpg.items()}, **{("grp", p): v for p, v in grp.items()}}
    spots, looked = [], 0
    for team, m in teams.items():
        opp = next(x for x in m.split("@") if x != team)
        roster = [p for p, t in team_now.items() if t == team and mpg.get(p, 0) >= ROT_MPG]
        out_t = {p for p in roster if status.get(p, ("", 0, ""))[1] >= OUT_P} | (extra_out or {}).get(team, set())
        unc = {p: status[p][1] for p in roster if p in status and SAFE_P <= status[p][1] < OUT_P and p not in out_t}
        cands = cur.filter(pl.col("team") == team).group_by("player_id").agg(pl.col("minutes").tail(5).mean().alias("m5"))
        for pid, m5 in cands.iter_rows():
            if team_now.get(pid) != team or m5 < CAND_MPG or pid in out_t:
                continue
            own_p = status.get(pid, ("", 0, ""))[1]
            if own_p >= SAFE_P:
                continue
            if last_ppg.get(pid, 0) >= STAR_PPG:
                continue
            looked += 1
            g = gt.filter(pl.col("player_id") == pid, pl.col("team") == team)
            last = trk_last.filter(pl.col("player_id") == pid)
            row = gf.filter(pl.col("opponent") == opp, pl.col("grp") == grp.get(pid, "W"))
            opp_f = {s: row[s][0] for s in ("pts", "reb", "ast")} if row.height else {}
            gl = gt_last.filter(pl.col("player_id") == pid, pl.col("team") == team)
            a = analyse(pid, pid_name.get(pid, str(pid)), team, opp, g, last, out_t, unc, id_name,
                        ca.get(pid, 0.55), cr.get(pid, 0.55), opp_f, spreads.get(team), gl)
            if a is None or not {x["kind"] for x in a["sits"]} & PRIMARY:
                continue  # a 3-game volume spike alone regressed (replay: -40%); it only supports a real spot
            a["opp_line"] = weakness_line(opp, grp.get(pid, "W"), day)
            a["unc"] = {pid_name.get(p, str(p)): v for p, v in unc.items()}
            c = max(CONF_MIN, confidence(a, sum(unc.values()), team in filed) - 0.10 * own_p)
            a["own_p"] = own_p
            sim = simulate(a)
            mk, books = market_probs(odds, a["name"])
            bs = bets_for(a, sim, mk, books, c)
            a.update(conf=c, sim_means={k: float(v.mean()) for k, v in sim.items()},
                     min_p10_90=(float(np.percentile(sim["minutes"], 10)), float(np.percentile(sim["minutes"], 90))), bets=bs)
            kinds = {x["kind"] + ("-vol" if x["kind"] == "emerging" and x.get("stat") in ("ast", "reb") else "") for x in a["sits"]}
            for b in bs:
                b["grade"] = grade(b["ev"], c, kinds, b.get("outlier", False))
            if any(not b.get("alt") for b in bs):
                spots.append(a)
    spots.sort(key=lambda a: -max(b["ev"] * a["conf"] for b in a["bets"]))
    return {"day": day, "spots": spots, "report_time": str(rep["report_time"].max()), "filed": filed, "teams": set(teams),
            "snapshot": odds["snapshot"].max() if odds.height else None, "looked": looked, "status": status,
            "pid_name": pid_name, "team_now": team_now, "mpg": mpg}


def conditional(day: date, base: dict) -> list[dict]:
    """For each Questionable rotation player (P(sits) 25-70%): spots that only exist if he sits."""
    out = []
    have = {(a["name"], b["bet"]) for a in base["spots"] for b in a["bets"]}
    for pid, (st, p, reason) in base["status"].items():
        team = base["team_now"].get(pid)
        if team not in base["teams"] or not (0.15 <= p < OUT_P) or base["mpg"].get(pid, 0) < KEY_MPG:
            continue
        alt = build(day, extra_out={team: {pid}})
        new = [(a, b) for a in alt["spots"] if a["team"] == team for b in a["bets"]
               if (a["name"], b["bet"]) not in have and not b.get("alt") and not b.get("outlier") and b["ev"] >= MIN_EV]
        out.append({"player": base["pid_name"].get(pid, str(pid)), "team": team, "p": p, "reason": reason,
                    "mpg": base["mpg"].get(pid, 0.0), "bets": sorted(new, key=lambda ab: -ab[1]["ev"])[:3]})
    out.sort(key=lambda c: -c["mpg"])
    return out


# ---------------------------------------------------------------- render
def _fmt_bet(b):
    est = "est. " if b["kind"] == "SGM" else ""
    return (f"**{b['bet']}** ({b['book'].replace('_au', '')}) {est}${b['price']:.2f} · fair ${1 / b['p']:.2f} · "
            f"EV {b['ev']:+.0%} · take at **${b['take_at']:.2f}**")


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
    ok = lambda b: not b.get("outlier") and not b.get("alt")
    card = [(a, next(b for b in a["bets"] if ok(b))) for a in res["spots"] if any(ok(b) for b in a["bets"])]
    L += ["", "## The card", "", "| Grade | Stake | Player | Bet | SGM version | Why |", "|---|---|---|---|---|---|"]
    for a, b in sorted(card, key=lambda ab: ("ABC".index(ab[1]["grade"]), -ab[1]["ev"]))[:CARD_N]:
        sg = next((x for x in a["bets"] if x["kind"] == "SGM" and x is not b), None)
        L.append(f"| **{b['grade']}** | {STAKE[b['grade']]} | **{a['name']}** ({a['team']} vs {a['opp']}) | {_fmt_bet(b)} | "
                 f"{_fmt_bet(sg) if sg else '—'} | {_why(a, b)}" + (f"; {a['opp_line']}" if a.get('opp_line') else "") + " |")
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
        if a["sample"] is not None and a["sample"].height:
            rows = a["sample"].tail(6).select("game_date", "opponent", "minutes", "pts", "reb", "ast", "potential_ast", "reb_chances").rows()
            L.append("- **games that match tonight**: " + "; ".join(
                f"{d:%d %b} vs {o}: {mi} min {p}/{r}/{x}" + (f" ({pa} pot, {rc} ch)" if pa is not None else "")
                for d, o, mi, p, r, x, pa, rc in rows))
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
    for a in res["spots"]:
        for i, b in enumerate(a["bets"]):
            rows.append({"day": res["day"], "player": a["name"], "team": a["team"], "kind": b["kind"], "bet": b["bet"],
                         "book": b["book"], "price": b["price"], "p": b["p"], "p_model": b["p_model"], "p_mkt": b["p_mkt"],
                         "ev": b["ev"], "grade": b["grade"], "conf": a["conf"], "card": i == 0 and not b.get("alt"), "alt": bool(b.get("alt")),
                         "situations": "; ".join(x["kind"] for x in a["sits"])})
    out = config.DATA_PROCESSED / "spots" / f"{res['day']}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)


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
        if b is None:
            hits.append(None)
            continue
        if r["kind"] == "SGM":
            legs = [(leg[x[-1]], int(x[:-1])) for x in r["bet"].split("/")]
        else:
            k, st = r["bet"].split("+ ")
            legs = [(st, int(k))]
        hits.append(all(sum(b[s] for s in st.split("+")) >= k for st, k in legs))
    return df.with_columns(pl.Series("hit", hits, dtype=pl.Boolean))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--no-conditional", action="store_true")
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
    res = build(args.date)
    cond = [] if args.no_conditional else conditional(args.date, res)
    md = render(res, cond)
    out = config.ROOT / "docs" / "slates" / f"{args.date.isoformat()}-spots.md"
    out.write_text(md)
    from nba_ladders.spots_html import render_html

    out.with_suffix(".html").write_text(render_html(res, cond))
    save(res, cond)
    print(md[:3000])
    print(f"wrote {out} and .html")


if __name__ == "__main__":
    main()
