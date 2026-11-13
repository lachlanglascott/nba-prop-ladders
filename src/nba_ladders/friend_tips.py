"""Friend's tips on the slate: their trend-first model (friend_model/, a port of their handoff), kept only where
it backtests, with our view on each.

Their trigger: over the last 2 / 3 games a player's chances (rebound chances, potential assists, 3PA) run
>= +12% / +10% over his season, and the opponent ranks bottom half for that stat. They then take the Sportsbet
rung >= $3 nearest the projected mean.

Backtest of that trigger, as-at, on Sportsbet $3+ rungs 2024-26 (player-games, 90% CI, by-season):
  assists:   +0.9% (-6 to +8%) vs -14.4% for all assist tails; 24-25 -0.9%, 25-26 +2.0%
             minutes flat (L3 < 1.05 x season): +8.1% (-3 to +19%); 24-25 +18%, 25-26 +2%
             with angle H ast >= 1.02: +5.1% (-8 to +18%)
  rebounds:  -20.6% (-28 to -13%), same as all rebound tails: the trend is noise. Dropped.
  threes:    not in the market backtest. Listed as untested, not bets.
So: assist trends only, 0.25u, minutes-flat ones first. A recency angle, not proven (CIs include 0).

    uv run python -m nba_ladders.friend_tips --date YYYY-MM-DD
"""

import argparse
import importlib.util
from datetime import date
from functools import lru_cache

from nba_ladders import config

# Honest chance = market's real chance x the trigger's backtested lift over the market (ROI ratio vs all assist tails,
# -14.4%): minutes flat (1 + 8.1%) / (1 - 14.4%) = 1.26; minutes up too (1 - 5.5%) / (1 - 14.4%) = 1.10.
LIFT = {True: 1.26, False: 1.10}
MIN_EV = 0.04
# threes price screen: their model 1.15-2x the market's real chance (their handoff: > 2x = likely model error), $2.50-10, top 6
THREE_GAP, THREE_GAP_MAX, THREE_PRICE, THREE_N = 1.15, 2.0, (2.5, 10.0), 6
FLAT_MIN = 1.05  # L3 minutes / season below this = minutes flat (the +8% subset)
STAKE = "0.25u"
LAB = {"A": "ast", "R": "reb", "M": "3pm"}


@lru_cache(maxsize=1)
def _fm():
    spec = importlib.util.spec_from_file_location("friend_run", config.ROOT / "friend_model" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tips(day: date) -> dict:
    """{"bets": [...assist trends], "ideas": [...rebound and threes trends, not bets]} for the slate's Friend's tips."""
    from nba_ladders.opp_position import tonight as opp_tonight
    from nba_ladders.sources.injury_report import player_key
    from nba_ladders.spots import market_ratio

    fm = _fm()
    res, cand, spr, _opp, _q = fm.model(day)
    lad = fm.ladders(day)
    opp_h = opp_tonight(day)
    grp_of = {"G": "G", "F": "W", "C": "B"}
    out = {"bets": [], "ideas": []}
    for stt, n, why in cand:
        r = res.get(n)
        if r is None:
            continue
        rungs = [(k, pr) for k, pr in lad.get(player_key(n), {}).get(stt, {}).items() if pr >= 3]
        if r[stt] is None or not rungs:
            continue
        k, pr = min(rungs, key=lambda kp: (abs(kp[0] - r[stt]), -kp[1]))
        q = fm.nb_tail(r[stt], k, fm.RR[stt])
        real = market_ratio(pr) / pr
        w = r["w"]
        m3, ms = (w.get(3) or {}).get("minutes"), (w.get(0) or {}).get("minutes")
        flat = bool(m3 and ms and m3 / ms < FLAT_MIN)
        # assists: the backtested lift; rebounds / threes: no tested lift, so the honest chance is the market's own
        p = min(real * LIFT[flat], 0.95) if stt == "A" else real
        opp_f = opp_h.get((r["o"], grp_of.get(r["ps"], "W")), {})
        f_ast = opp_f.get("ast", 1.0)
        b = {"name": n, "team": r["t"], "opp": r["o"], "kind": "single", "bet": f"{k}+ {LAB[stt]}", "book": "sportsbet",
             "price": pr, "p_model": q, "p_mkt": real, "p": p, "ev": pr * p - 1, "take_at": 1.10 / p, "mu": r[stt],
             "flat": flat, "opp_ast": f_ast, "opp_reb": opp_f.get("reb", 1.0), "spread": spr.get(r["t"], 0.0), "q_flag": r["q"],
             "why": why, "stat": stt, "gap": q / real if real else None}
        if stt != "A":
            b["bet_ok"] = False
            out["ideas"].append(b)
            continue
        b["bet_ok"] = flat and abs(b["spread"]) < 15 and not b["q_flag"] and b["ev"] >= MIN_EV
        out["bets"].append(b)
    # minutes-flat (the backtested best subset) first, then angle H, then their edge
    # threes price screen: every Sportsbet threes rung where their model beats the market's real chance by THREE_GAP+
    # (not only trend-flagged players), so threes ideas show up on quiet trend nights too
    have = {(b["name"], b["bet"]) for b in out["ideas"]}
    keyed = {player_key(n): r for n, r in res.items()}
    screen = []
    for pk, L in lad.items():
        r = keyed.get(pk)
        if r is None or r.get("M") is None:
            continue
        for k, pr in L.get("M", {}).items():
            if not (THREE_PRICE[0] <= pr <= THREE_PRICE[1]):
                continue
            q = fm.nb_tail(r["M"], k, fm.RR["M"])
            real = market_ratio(pr) / pr
            if not (THREE_GAP <= q / real <= THREE_GAP_MAX) or (r["name"], f"{k}+ 3pm") in have:
                continue
            w = r["w"]
            m3, ms = (w.get(3) or {}).get("minutes"), (w.get(0) or {}).get("minutes")
            tri = lambda c, w=w: "/".join(f"{(w.get(i) or {}).get(c) or 0:.1f}" for i in (1, 2, 3, 0))
            screen.append({"name": r["name"], "team": r["t"], "opp": r["o"], "kind": "single", "bet": f"{k}+ 3pm", "book": "sportsbet",
                           "price": pr, "p_model": q, "p_mkt": real, "p": real, "ev": pr * real - 1, "take_at": 1.10 / real, "mu": r["M"],
                           "flat": bool(m3 and ms and m3 / ms < FLAT_MIN), "opp_ast": 1.0, "opp_reb": 1.0, "spread": spr.get(r["t"], 0.0),
                           "q_flag": r["q"], "why": f"price screen · 3PA {tri('fg3a')} · 3PM {tri('fg3m')} (L1/L2/L3/S)", "stat": "M",
                           "gap": q / real, "bet_ok": False, "screen": True})
    best = {}
    for b in sorted(screen, key=lambda b: -b["gap"]):
        best.setdefault(b["name"], b)  # one rung per player: the biggest gap
    out["ideas"] += sorted(best.values(), key=lambda b: -b["gap"])[:THREE_N]
    out["bets"].sort(key=lambda b: (not b["bet_ok"], -(b["opp_ast"] >= 1.02), -b["ev"]))
    out["ideas"].sort(key=lambda b: (b["stat"], -(b["gap"] or 0)))
    return out


IDEA_TAG = {"R": "rebound trend (backtest −21%: idea only)", "M": "threes (untested: idea only)"}


def idea_note(b: dict) -> str:
    bits = [IDEA_TAG[b["stat"]], f"their model {b['gap']:.1f}× the market" if b.get("gap") else ""]
    if b["stat"] == "R":
        bits.append(f"angle H opp reb {b['opp_reb']:.2f}" + (" (good)" if b["opp_reb"] >= 1.02 else " (bad)" if b["opp_reb"] < 0.98 else ""))
    bits.append("minutes flat" if b["flat"] else "minutes up")
    if abs(b["spread"]) >= 9.5:
        bits.append(f"blowout risk (spread {b['spread']:+.1f})")
    if b["q_flag"]:
        bits.append("questionable")
    return " · ".join(x for x in bits if x)


def note(b: dict) -> str:
    bits = ["minutes flat ★" if b["flat"] else "minutes up too (weaker: +8% vs -6% in backtest)"]
    if b["opp_ast"] >= 1.02:
        bits.append(f"angle H agrees (opp ast {b['opp_ast']:.2f})")
    elif b["opp_ast"] < 0.97:
        bits.append(f"angle H disagrees (opp ast {b['opp_ast']:.2f})")
    if abs(b["spread"]) >= 9.5:
        bits.append(f"blowout risk (spread {b['spread']:+.1f})")
    if b["q_flag"]:
        bits.append("questionable: wait for status")
    return " · ".join(bits)


def render_md(ft: dict) -> list[str]:
    if not (ft["bets"] or ft.get("ideas")):
        return []
    L = ["", "## Friend's tips (recency: assist trends, 0.25u)", "",
         ("Their trend-first model, kept where it backtests: assist trends with minutes flat (+8% at Sportsbet 2024-26, CI "
         "−3 to +19%). Rebound trends dropped (−21%). Honest fair = market's real chance × the trigger's "
         "backtested lift (×1.26 minutes flat, ×1.10 otherwise), not their model."), "",
         "| | Player | Bet | Their model | Honest fair | EV | Trend | Notes |", "|---|---|---|---|---|---|---|---|"]
    for b in ft["bets"]:
        L.append(f"| {'**Bet**' if b['bet_ok'] else 'Watch'} | **{b['name']}** ({b['team']} vs {b['opp']}) | **{b['bet']}** (sportsbet) ${b['price']:.2f} · "
                 f"take at **${b['take_at']:.2f}** | {b['p_model']:.0%} (exp {b['mu']:.1f}) | ${1 / b['p']:.2f} | {b['ev']:+.0%} | {b['why']} | {note(b)} |")
    if ft.get("ideas"):
        L += ["", ("**Ideas: their rebound trends and threes** (trends plus a threes price screen; not bets: rebound trends lost 21% "
              "in the backtest, threes untested; shown for leads and graded so we learn). Fair = the market's own chance."), "",
              "| Player | Bet | Their model | Market fair | Trend | Notes |", "|---|---|---|---|---|---|"]
        for b in ft["ideas"]:
            L.append(f"| **{b['name']}** ({b['team']} vs {b['opp']}) | {b['bet']} (sportsbet) ${b['price']:.2f} | {b['p_model']:.0%} (exp {b['mu']:.1f}) | "
                     f"${1 / b['p_mkt']:.2f} | {b['why']} | {idea_note(b)} |")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    print("\n".join(render_md(tips(ap.parse_args().date))))


if __name__ == "__main__":
    main()
