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
    """{"bets": [...], "untested": [...], "dropped": [...]} for the slate's Friend's tips section."""
    from nba_ladders.opp_position import tonight as opp_tonight
    from nba_ladders.sources.injury_report import player_key
    from nba_ladders.spots import market_ratio

    fm = _fm()
    res, cand, spr, _opp, _q = fm.model(day)
    lad = fm.ladders(day)
    opp_h = opp_tonight(day)
    grp_of = {"G": "G", "F": "W", "C": "B"}
    out = {"bets": [], "untested": [], "dropped": []}
    for stt, n, why in cand:
        r = res.get(n)
        if r is None:
            continue
        if stt == "R":
            out["dropped"].append(n)
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
        p = min(real * LIFT[flat], 0.95)
        f_ast = opp_h.get((r["o"], grp_of.get(r["ps"], "W")), {}).get("ast", 1.0)
        b = {"name": n, "team": r["t"], "opp": r["o"], "kind": "single", "bet": f"{k}+ {LAB[stt]}", "book": "sportsbet",
             "price": pr, "p_model": q, "p_mkt": real, "p": p, "ev": pr * p - 1, "take_at": 1.10 / p, "mu": r[stt],
             "flat": flat, "opp_ast": f_ast, "spread": spr.get(r["t"], 0.0), "q_flag": r["q"], "why": why}
        if stt == "M":
            out["untested"].append(b)
            continue
        b["bet_ok"] = flat and abs(b["spread"]) < 15 and not b["q_flag"] and b["ev"] >= MIN_EV
        out["bets"].append(b)
    # minutes-flat (the backtested best subset) first, then angle H, then their edge
    out["bets"].sort(key=lambda b: (not b["bet_ok"], -(b["opp_ast"] >= 1.02), -b["ev"]))
    return out


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
    if not (ft["bets"] or ft["untested"] or ft["dropped"]):
        return []
    L = ["", "## Friend's tips (recency: assist trends, 0.25u)", "",
         ("Their trend-first model, kept where it backtests: assist trends with minutes flat (+8% at Sportsbet 2024-26, CI "
         "−3 to +19%). Rebound trends dropped (−21%). Honest fair = market's real chance × the trigger's "
         "backtested lift (×1.26 minutes flat, ×1.10 otherwise), not their model."), "",
         "| | Player | Bet | Their model | Honest fair | EV | Trend | Notes |", "|---|---|---|---|---|---|---|---|"]
    for b in ft["bets"]:
        L.append(f"| {'**Bet**' if b['bet_ok'] else 'Watch'} | **{b['name']}** ({b['team']} vs {b['opp']}) | **{b['bet']}** (sportsbet) ${b['price']:.2f} · "
                 f"take at **${b['take_at']:.2f}** | {b['p_model']:.0%} (exp {b['mu']:.1f}) | ${1 / b['p']:.2f} | {b['ev']:+.0%} | {b['why']} | {note(b)} |")
    if ft["untested"]:
        L.append("")
        L.append("Untested (threes aren't in the market backtest): " + "; ".join(
            f"{b['name']} {b['bet']} ${b['price']:.2f} (their {b['p_model']:.0%})" for b in ft["untested"]))
    if ft["dropped"]:
        L.append("")
        L.append("Rebound trends dropped (backtest −21%): " + ", ".join(sorted(set(ft["dropped"]))))
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    print("\n".join(render_md(tips(ap.parse_args().date))))


if __name__ == "__main__":
    main()
