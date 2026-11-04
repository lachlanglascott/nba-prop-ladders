# ruff: noqa: B023  (closures are called inside the loop that defines them)
"""Friend's NBA alt-ladder model, ported from the handoff doc (see README). Separate from ours."""

import argparse
import math
from datetime import date, datetime, timedelta

import polars as pl
from scipy import stats as st

from nba_ladders import config
from nba_ladders.market import TEAM_ABBR, latest
from nba_ladders.scenarios import positions
from nba_ladders.sources.injury_report import ET, player_key, report_for_date
from nba_ladders.sources.odds_api import RAW_DIR, load_cached, parse_spreads

SEASON = "22026"
RR = {"A": 12.5, "R": 20.0, "M": 6.667}  # NB dispersion (handoff 4.4)
W, W3 = (0.15, 0.45, 0.40), (0.15, 0.50, 0.35)


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def nb_tail(mu, k, r):
    """P(X >= k), NB with mean mu and size r (handoff: 1 - sum_{i<k} NB(i))."""
    if mu <= 0:
        return 0.0
    p = r / (r + mu)
    return float(1 - st.nbinom.cdf(k - 1, r, p))


def aud(q, pr):  # handoff 4.5
    return 0.35 * q + 0.65 * (0.88 / pr)


# ------------------------------------------------------------------ data as at the day
def load(day):
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(pl.col("season_id") == SEASON, pl.col("game_date") < day)
    trk = pl.read_parquet(config.DATA_PROCESSED / "tracking.parquet").select("player_id", "game_date", "potential_ast", "reb_chances")
    pg = pg.join(trk, on=["player_id", "game_date"], how="left")
    pos = positions().with_columns(pl.col("position").str.slice(0, 1).alias("ps")).select("player_id", "ps")
    return pg, pos


def windows(pg):
    """Per player: L1/L2/L3 (last n TEAM games of his current team) and S (season) per-game averages."""
    last_team = pg.sort("game_date").group_by("player_id").agg(pl.col("team").last().alias("t"), pl.col("player_name").last())
    tg = pg.select("team", "game_id", "game_date").unique().sort("game_date")
    out = {}
    cols = ["minutes", "pts", "reb", "ast", "fg3a", "fg3m", "potential_ast", "reb_chances"]
    for pid, t, name in last_team.iter_rows():
        mine = pg.filter(pl.col("player_id") == pid)
        team_games = tg.filter(pl.col("team") == t)["game_id"].to_list()
        w = {}
        for n in (1, 2, 3):
            ids = team_games[-n:]
            g = mine.filter(pl.col("game_id").is_in(ids))
            w[n] = None if g.is_empty() else {c: (g[c].mean() if g[c].null_count() < g.height else None) for c in cols} | {"gp": g.height}
        w[0] = {c: (mine[c].mean() if mine[c].null_count() < mine.height else None) for c in cols} | {"gp": mine.height}
        out[name] = {"pid": pid, "t": t, "w": w}
    return out


def opp_ranks(pg, pos):
    """Rank 1-30 per opponent (30 = allows most) of AST / REB / 3PM allowed, overall and by position; MISS; P3%."""
    d = pg.join(pos, on="player_id", how="left").with_columns(pl.col("ps").fill_null("F"))
    def per_game(df):
        return df.group_by("opponent", "game_id").agg(pl.col("ast").sum(), pl.col("reb").sum(), pl.col("fg3m").sum(),
                                                       pl.col("fg3a").sum(), (pl.col("fga") - pl.col("fgm")).sum().alias("miss")) \
                 .group_by("opponent").agg(pl.col("ast").mean().alias("A"), pl.col("reb").mean().alias("R"), pl.col("fg3m").mean().alias("M"),
                                           pl.col("miss").mean().alias("MISS"), (pl.col("fg3m").sum() / pl.col("fg3a").sum()).alias("P3"))
    def rank(df, col):
        s = df.sort(col)
        return {r: i + 1 for i, r in enumerate(s["opponent"].to_list())}
    OR = {}
    allv = per_game(d)
    OR["ALL"] = {k: rank(allv, k) for k in ("A", "R", "M", "MISS", "P3")}
    for ps in ("G", "F", "C"):
        v = per_game(d.filter(pl.col("ps") == ps))
        OR[ps] = {k: rank(v, k) for k in ("A", "R", "M")}
    return OR


def pace(pg):
    t = pg.group_by("game_id", "team").agg((pl.col("fga").sum() + 0.44 * pl.col("fta").sum() - pl.col("oreb").sum() + pl.col("tov").sum()).alias("poss"),
                                           pl.col("minutes").sum().alias("tm"))
    g = t.group_by("game_id").agg(pl.col("poss").mean().alias("gp"), pl.col("tm").max().alias("mx"))
    t = t.join(g, on="game_id").with_columns((pl.col("gp") * 240 / pl.col("mx").clip(lower_bound=200)).alias("pace"))
    return dict(t.group_by("team").agg(pl.col("pace").mean()).iter_rows())


def slate(day):
    rep = report_for_date(day).filter(pl.col("game_date") == day).with_columns(
        pl.col("team_name").replace_strict({k.replace(" ", ""): v for k, v in TEAM_ABBR.items()}, default=None).alias("team"))
    games = rep["matchup"].drop_nulls().unique().to_list()
    OPP = {}
    for m in games:
        a, h = m.split("@")
        OPP[a], OPP[h] = h, a
    names = lambda s: set(rep.filter(pl.col("status").is_in(s), pl.col("player").is_not_null())["player_key"].to_list())
    OUT, Q = names(["Out", "Doubtful"]), names(["Questionable"])
    SPR = {}
    for f in sorted((RAW_DIR / "spreads").glob("*.json")):
        import json

        for r in parse_spreads(json.loads(f.read_text())).iter_rows(named=True):
            if datetime.fromisoformat(r["commence_time"]).astimezone(ET).date() == day and TEAM_ABBR.get(r["team_name"]) in OPP:
                SPR[TEAM_ABBR[r["team_name"]]] = r["spread"]
    return OPP, OUT, Q, SPR, rep


def ladders(day):
    o = latest(load_cached("live"))
    lo = datetime(day.year, day.month, day.day, 6, tzinfo=ET)
    o = o.filter(pl.col("commence_time").map_elements(lambda s: lo <= datetime.fromisoformat(s) < lo + timedelta(days=1), return_dtype=pl.Boolean),
                 pl.col("bookmaker") == "sportsbet", pl.col("side") == "Over", pl.col("stat").is_in(["ast", "reb", "3pm"]), pl.col("threshold").is_not_null())
    lad = {}
    for p, s, k, pr in o.select("player", "stat", "threshold", "price").iter_rows():
        lad.setdefault(player_key(p), {}).setdefault({"ast": "A", "reb": "R", "3pm": "M"}[s], {})[k] = pr
    return lad


# ------------------------------------------------------------------ model (handoff 5.4)
def model(day):
    pg, pos = load(day)
    T = windows(pg)
    OR = opp_ranks(pg, pos)
    PACE = pace(pg)
    OPP, OUT, Q, SPR, _rep = slate(day)
    POS = dict(pos.iter_rows())
    played_yday = set(pg.filter(pl.col("game_date") == day - timedelta(days=1))["team"].unique().to_list())
    RET = {}
    for n, p in T.items():
        if p["t"] in OPP and p["w"][1] is None and (p["w"][0]["minutes"] or 0) > 24 and player_key(n) not in OUT:
            RET.setdefault(p["t"], []).append(n)
    v = lambda p, k, i: (p["w"][i] or {}).get(k)
    def blend(p, k, w):
        S = v(p, k, 0)
        if S is None:
            return None
        L3 = v(p, k, 3) if v(p, k, 3) is not None else S
        L1 = v(p, k, 1) if v(p, k, 1) is not None else L3
        return w[0] * L1 + w[1] * L3 + w[2] * S
    res, cand = {}, []
    for n, p in T.items():
        t = p["t"]
        o = OPP.get(t)
        S0 = p["w"][0]
        if o is None or player_key(n) in OUT or S0["gp"] < 4 or (S0["minutes"] or 0) < 18:
            continue
        ret_me = n in RET.get(t, [])
        team_ret = bool(set(RET.get(t, [])) - {n})
        S = S0["minutes"]
        m = 0.45 * S + 0.35 * (v(p, "minutes", 3) or S) + 0.2 * (v(p, "minutes", 1) or S)
        if ret_me:
            m = 0.7 * S + 0.3 * m
        s = SPR.get(t, 0.0)
        pb = 1 - ncdf((18 + s) / 13.5) + ncdf((-18 + s) / 13.5)
        starter = S >= 27
        m = m - (pb * 9 if starter else -pb * 4)
        if starter and t in played_yday:
            m -= 1.5  # back-to-back (handoff 4.2)
        bm = (0.7 * S + 0.3 * blend(p, "minutes", W)) if ret_me else blend(p, "minutes", W)
        bm3 = (0.7 * S + 0.3 * blend(p, "minutes", W3)) if ret_me else blend(p, "minutes", W3)
        f, f3 = m / bm, m / bm3
        mt = lambda k: 1 + (OR["ALL"][k].get(o, 15.5) - 15.5) / 15.5 * 0.06
        pf = ((PACE.get(t, 100) + PACE.get(o, 100)) / 2) / 101.5
        def rv(k, w):
            b = blend(p, k, w)
            if b is not None and team_ret and v(p, k, 3) is not None:
                b = 0.7 * v(p, k, 0) + 0.3 * b
            return b
        mA = mR = m3 = None
        Sp, Sc, Sa = v(p, "potential_ast", 0), v(p, "reb_chances", 0), v(p, "fg3a", 0)
        if Sp:
            c = 0.65 * (S0["ast"] / max(Sp, 1)) + 0.35 * ((v(p, "ast", 3) or 0) / max(v(p, "potential_ast", 3) or 1, 1))
            c = (c * 8 + 0.55 * 2) / 10
            mA = rv("potential_ast", W) * c * f * mt("A") * pf
        if Sc:
            c = 0.65 * (S0["reb"] / max(Sc, 1)) + 0.35 * ((v(p, "reb", 3) or 0) / max(v(p, "reb_chances", 3) or 1, 1))
            c = (c * 8 + 0.55 * 2) / 10
            mR = rv("reb_chances", W) * c * f * mt("R") * pf
        if Sa:
            pc = S0["fg3m"] / max(Sa, 0.1)
            nn = Sa * S0["gp"]
            pc = (pc * nn + 0.36 * 80) / (nn + 80)
            m3 = rv("fg3a", W3) * pc * f3 * mt("M") * pf
        res[n] = {"name": n, "t": t, "o": o, "min": m, "pb": pb, "A": mA, "R": mR, "M": m3, "ret": team_ret, "q": player_key(n) in Q,
                  "ps": POS.get(p["pid"], "F"), "w": p["w"]}
        # trend-first candidates
        if p["w"][1] is None or S < 20:
            continue
        ps = res[n]["ps"]
        def tr(k):
            S1, L2, L3 = v(p, k, 0), v(p, k, 2), v(p, k, 3)
            if not S1 or L2 is None or L3 is None:
                return None
            return {"r2": L2 / S1, "r3": L3 / S1, "line": "/".join(f"{x:.1f}" if x is not None else "-" for x in (v(p, k, 1), L2, L3, S1))}
        rc, pa, ta = tr("reb_chances"), tr("potential_ast"), tr("fg3a")
        rk = lambda key: max(OR.get(ps, OR["ALL"])[key].get(o, 0), OR["ALL"][key].get(o, 0))
        if rc and (Sc or 0) >= 8 and rc["r2"] >= 1.12 and rc["r3"] >= 1.1 and rk("R") >= 18:
            cand.append(("R", n, f"chances {rc['line']} (L1/L2/L3/S) · opp reb rank {OR['ALL']['R'].get(o)}/{OR.get(ps, OR['ALL'])['R'].get(o)} ({ps})"))
        if pa and (Sp or 0) >= 6 and pa["r2"] >= 1.12 and pa["r3"] >= 1.1 and rk("A") >= 18:
            cand.append(("A", n, f"potentials {pa['line']} · opp ast rank {OR['ALL']['A'].get(o)}/{OR.get(ps, OR['ALL'])['A'].get(o)} ({ps})"))
        if ta and (Sa or 0) >= 4.5 and ta["r3"] >= 0.95:
            s_pct = S0["fg3m"] / Sa
            l3 = (v(p, "fg3m", 3) or 0) / max(v(p, "fg3a", 3) or 1, 1)
            if l3 <= s_pct - 0.03 and rk("M") >= 18:
                cand.append(("M", n, f"3PA {ta['line']} · L3 {l3:.0%} vs S {s_pct:.0%} · opp 3PM rank {OR['ALL']['M'].get(o)}"))
    return res, cand, SPR, OPP, Q


# ------------------------------------------------------------------ price screen (5.6) + rules (3, 6)
def card(day):
    res, cand, SPR, _OPP, _Q = model(day)
    lad = ladders(day)
    keyed = {player_key(n): r for n, r in res.items()}
    singles, unmatched = [], []
    for pk, L in lad.items():
        r = keyed.get(pk)
        if r is None:
            unmatched.append(pk)
            continue
        for stt, rungs in L.items():
            mu = r[stt]
            if mu is None:
                continue
            for k, pr in rungs.items():
                if pr < 3:
                    continue
                q = nb_tail(mu, k, RR[stt])
                a = aud(q, pr)
                e = a * pr - 1
                if e < 0.06:
                    continue
                singles.append({"name": r["name"], "t": r["t"], "o": r["o"], "stat": stt, "k": k, "price": pr, "q": q, "a": a, "e": e,
                                "gap": q / (0.88 / pr), "min_price": 1.1 / a, "mu": mu, "q_flag": r["q"]})
    # trend-first (handoff 4.7): for each trend candidate take the Sportsbet rung >= $3 nearest the projected
    # mean, compute the audited edge, require price >= minimum (1.1 / audited p) and gap <= 2, then caps
    picks = []
    for stt, n, why in cand:
        r = res.get(n)
        L = lad.get(player_key(n), {}).get(stt, {})
        rungs = [(k, pr) for k, pr in L.items() if pr >= 3]
        if r is None or r[stt] is None or not rungs:
            continue
        k, pr = min(rungs, key=lambda kp: (abs(kp[0] - r[stt]), -kp[1]))
        q = nb_tail(r[stt], k, RR[stt])
        a = aud(q, pr)
        picks.append({"name": n, "t": r["t"], "o": r["o"], "stat": stt, "k": k, "price": pr, "q": q, "a": a, "e": a * pr - 1,
                      "gap": q / (0.88 / pr), "min_price": 1.1 / a, "mu": r[stt], "q_flag": r["q"], "trend": why})
    core, watch, per_game, used = [], [], {}, set()
    for b in sorted(picks, key=lambda b: -b["e"]):
        spread = abs(SPR.get(b["t"], 0.0))
        b["blowout"] = "skip" if spread >= 15 else "watch" if spread >= 9.5 else ""
        ok = b["price"] >= b["min_price"] and b["gap"] <= 2 and b["blowout"] != "skip"
        g = "@".join(sorted([b["t"], b["o"]]))
        if not ok:
            b["why_not"] = ("price below minimum" if b["price"] < b["min_price"] else "gap > 2 (likely model error)" if b["gap"] > 2
                            else "blowout skip")
            watch.append(b)
            continue
        if b["name"] in used or per_game.get(g, 0) >= 2 or len(core) >= 5 or b["blowout"] == "watch" or b["q_flag"]:
            b["why_not"] = ("questionable: wait for status" if b["q_flag"] else "blowout watch (spread 9.5-15)" if b["blowout"] == "watch"
                            else "caps")
            watch.append(b)
            continue
        core.append(b)
        used.add(b["name"])
        per_game[g] = per_game.get(g, 0) + 1
    return {"core": core, "watch": watch, "singles": singles, "cand": cand, "unmatched": unmatched, "res": res}


def grade(day, c):
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(pl.col("game_date") == day)
    box = {player_key(r["player_name"]): r for r in pg.iter_rows(named=True)}
    col = {"A": "ast", "R": "reb", "M": "fg3m"}
    out = []
    for b in c:
        x = box.get(player_key(b["name"]))
        out.append({**b, "actual": None if x is None else x[col[b["stat"]]], "hit": None if x is None else x[col[b["stat"]]] >= b["k"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    ap.add_argument("--grade", action="store_true")
    a = ap.parse_args()
    c = card(a.date)
    lab = {"A": "ast", "R": "reb", "M": "threes"}
    print(f"{len(c['res'])} players modelled · {len(c['cand'])} trend candidates · {len(c['singles'])} singles with edge >= 6%")
    print("\nCORE CARD (trend + edge + gap <= 2 + caps):")
    for b in c["core"]:
        print(f"  {b['name']} ({b['t']} v {b['o']}) {b['k']}+ {lab[b['stat']]} @ ${b['price']:.2f} | model {b['q']:.0%} audited {b['a']:.0%} "
              f"edge {b['e']:+.0%} min ${b['min_price']:.2f} gap x{b['gap']:.1f} | mu {b['mu']:.2f} | {b['trend']}{' | Q' if b['q_flag'] else ''}")
    print("\nWATCH / NOT BET:")
    for b in c["watch"]:
        print(f"  {b['name']} {b['k']}+ {lab[b['stat']]} @ ${b['price']:.2f} edge {b['e']:+.0%} min ${b['min_price']:.2f} gap x{b['gap']:.1f}: {b['why_not']}")
    if a.grade:
        g = grade(a.date, c["core"])
        played = [x for x in g if x["hit"] is not None]
        if played:
            pnl = sum(x["price"] * 0.25 if x["hit"] else 0 for x in played) - 0.25 * len(played)
            print(f"\nGRADE core: {sum(x['hit'] for x in played)}/{len(played)} hit, audited expected {sum(x['a'] for x in played):.2f}, "
                  f"P&L {pnl:+.2f}u at 0.25u")
        for x in g:
            print(f"  {x['name']} {x['k']}+ {lab[x['stat']]}: {x['actual']} -> {'W' if x['hit'] else 'L' if x['hit'] is not None else 'DNP'}")
        ga = [x for x in grade(a.date, c["singles"]) if x["hit"] is not None]
        if ga:
            print(f"all edge singles: {sum(x['hit'] for x in ga)}/{len(ga)} hit, audited exp {sum(x['a'] for x in ga):.1f}, "
                  f"flat 1u P&L {sum(x['price'] if x['hit'] else 0 for x in ga) - len(ga):+.1f}u")


if __name__ == "__main__":
    main()
