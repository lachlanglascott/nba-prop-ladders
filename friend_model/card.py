"""Friend's model: the 5-tip card as HTML (their look: night scoreboard, ticket picks, gold price,
green edge, amber risk). Picks follow the handoff: trend-first core bets, then price-screen fills
(singles and passer+shooter SGMs), all under the caps. One line per tip gives OUR view.

    uv run python friend_model/card.py --date 2026-11-04     # -> docs/slates/<date>-friend.html
"""

import argparse
import math
import sys
from datetime import date, datetime, timedelta
from html import escape as esc
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import run as F

from nba_ladders import config
from nba_ladders.market import latest
from nba_ladders.sources.injury_report import ET, player_key
from nba_ladders.sources.odds_api import load_cached

AEDT = ZoneInfo("Australia/Sydney")
FILL_MAX = 10.0  # their cards sit ~$4-10; longer price-screen rungs aren't used as fills
LAB = {"A": "assists", "R": "rebounds", "M": "threes"}
VOL = {"A": "potential assists", "R": "rebound chances", "M": "3PA"}


def ninv(p):
    """Inverse standard normal (bisection; fine for ranking)."""
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if F.ncdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def two_player_sgms(res, lad):
    """Handoff 5.6: passer's assists x a teammate's threes. Legs audited >= 25%, est 0.85 x legs, edge >= 10%."""
    out = []
    keyed = {player_key(n): r for n, r in res.items()}
    by_team = {}
    for pk, L in lad.items():
        r = keyed.get(pk)
        if r is not None:
            by_team.setdefault(r["t"], []).append((r, L))
    for rows in by_team.values():
        for A, LA in rows:
            for S, LS in rows:
                if A is S or A["A"] is None or S["M"] is None or "A" not in LA or "M" not in LS:
                    continue
                for ka, pa in LA["A"].items():
                    for ks, ps_ in LS["M"].items():
                        qa, qs = F.aud(F.nb_tail(A["A"], ka, F.RR["A"]), pa), F.aud(F.nb_tail(S["M"], ks, F.RR["M"]), ps_)
                        if qa < 0.25 or qs < 0.25:
                            continue
                        j = qa * qs + 0.25 * phi(ninv(1 - qa)) * phi(ninv(1 - qs))
                        est = 0.85 * pa * ps_
                        if not 3 <= est <= 20:
                            continue
                        e = j * est - 1
                        if e >= 0.10:
                            out.append({"kind": "sgm2", "name": A["name"], "mate": S["name"], "t": A["t"], "o": A["o"], "ka": ka, "ks": ks,
                                        "pa": pa, "ps": ps_, "price": est, "j": j, "e": e, "min_price": 1.1 / j, "q_flag": A["q"] or S["q"]})
    return sorted(out, key=lambda b: -b["e"])


def tips(day):
    c = F.card(day)
    res = c["res"]
    lad = F.ladders(day)
    SPR = F.slate(day)[3]
    tipped, per_game, used = [], {}, set()

    def room(b, names):
        g = "@".join(sorted([b["t"], b["o"]]))
        spread = abs(SPR.get(b["t"], 0.0))
        return not (spread >= 15 or any(n in used for n in names) or per_game.get(g, 0) >= 2 or len(tipped) >= 5)

    def take(b, tier, names):
        g = "@".join(sorted([b["t"], b["o"]]))
        b["tier"] = tier
        tipped.append(b)
        used.update(names)
        per_game[g] = per_game.get(g, 0) + 1

    for b in c["core"]:
        if room(b, [b["name"]]):
            take({**b, "kind": "single"}, "core", [b["name"]])
    # fills: price-screen singles that clear the minimum price and the gap check, then SGMs
    fills = [s for s in c["singles"] if s["price"] >= s["min_price"] and s["gap"] <= 2 and s["price"] <= FILL_MAX]
    for s in sorted(fills, key=lambda s: -s["e"]):
        spread = abs(SPR.get(s["t"], 0.0))
        if room(s, [s["name"]]) and not s["q_flag"] and spread < 9.5:
            s = {**s, "kind": "single", "trend": next((w for st_, n, w in c["cand"] if st_ == s["stat"] and n == s["name"]), "")}
            take(s, "small", [s["name"]])
    for s in two_player_sgms(res, lad):
        if room(s, [s["name"], s["mate"]]) and not s["q_flag"] and abs(SPR.get(s["t"], 0.0)) < 9.5:
            take(s, "small", [s["name"], s["mate"]])
    return tipped, c, SPR


OURS: dict = {}


def load_ours(day):
    """Our spots engine for the same day: player -> its bets (honest fair)."""
    from nba_ladders import spots as S

    res = S.build(day)
    for a in res["spots"] + res.get("leans", []):
        OURS[player_key(a["name"])] = a


def our_view(b):
    """Their audit redone with the market's REAL chance (alt overs hit 77-93% of implied, 664k Sportsbet rungs),
    plus whether our engine has a spot on the player."""
    from nba_ladders.spots import market_ratio

    if b["kind"] == "sgm2":
        return f"SGM: check the builder pays at least ${b['min_price']:.2f} (their joint {b['j']:.0%}); our engine doesn't price two-player SGMs."
    real = market_ratio(b["price"]) / b["price"]
    a2 = 0.35 * b["q"] + 0.65 * real
    e2 = a2 * b["price"] - 1
    ours = OURS.get(player_key(b["name"]))
    if ours is None:
        mine = "no spot in our engine (no lineup role change or bounce)"
    else:
        same = next((x for x in ours["bets"] if x["kind"] == "single" and x["bet"] == f"{b['k']}+ { {'A': 'ast', 'R': 'reb', 'M': '3pm'}[b['stat']]}"), None)
        mine = (f"our engine also has this bet: honest fair ${1 / same['p']:.2f}" if same
                else f"our engine has a different spot on him: {ours['bets'][0]['bet']}")
    verdict = "agree" if e2 >= 0.05 and ours is not None else "lean" if e2 >= 0 else "disagree"
    return (f"{verdict.upper()}. With the market's real chance ({real:.0%}, not 0.88/price = {0.88 / b['price']:.0%}) their audited "
            f"becomes {a2:.0%} → edge {e2:+.0%}; {mine}.")


def risks(b, res, spr):
    r = res.get(b["name"], {})
    out = []
    pb = r.get("pb", 0)
    if pb >= 0.12:
        out.append(f"Blowout chance {pb:.0%} (spread {spr.get(b['t'], 0):+.1f}): fourth-quarter minutes go")
    if b["kind"] == "single":
        w = r.get("w", {})
        k = {"A": "potential_ast", "R": "reb_chances", "M": "fg3a"}[b["stat"]]
        l1, s = (w.get(1) or {}).get(k), (w.get(0) or {}).get(k)
        if l1 and s and l1 >= 1.4 * s:
            out.append(f"The trend leans on the last game ({l1:.0f} {VOL[b['stat']]} vs {s:.1f} season)")
        if b.get("gap", 0) >= 1.6:
            out.append(f"Model is {b['gap']:.1f}× the market: if there's news, the market has it")
        out.append(f"Conversion: needs {b['k']}+ from about {b['mu']:.1f} expected")
    else:
        out.append(f"Two players must both land: {b['name']} {b['ka']}+ ast and {b['mate']} {b['ks']}+ threes")
        out.append("Builder price is an estimate: bet only at the minimum or better")
    if r.get("ret"):
        out.append("A teammate returning from absence takes usage")
    return out[:2]


CSS = """
:root{--bg:#0a0f17;--surface:#111925;--surface-2:#172131;--line:#1f2a3b;--fg:#eef3fa;--muted:#93a1b8;--faint:#5f6d84;
--price:#ffc53d;--edge:#3dff98;--edge-soft:#0c2a1c;--risk:#ffad42;--risk-soft:#2c1e0a;--neg:#ff6b6b;--neg-soft:#2c1314;--sel:#7aa8ff;--sel-soft:#16233f;
--display:"Barlow Condensed","Arial Narrow",system-ui,sans-serif;--body:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 var(--body);font-variant-numeric:tabular-nums}
main{max-width:1100px;margin:0 auto;padding:26px 16px 70px}
h1{font:800 clamp(34px,6vw,54px)/.95 var(--display);text-transform:uppercase;margin:6px 0 8px;letter-spacing:.01em}
.eyebrow{font:700 12px/1 var(--display);letter-spacing:.18em;text-transform:uppercase;color:var(--price)}
.lede{color:var(--muted);max-width:70ch;margin:0}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 4px}
.pill{font:600 12px/1 var(--body);padding:6px 10px;border-radius:999px;background:var(--surface-2);border:1px solid var(--line);color:var(--muted)}
.pill.warn{color:var(--risk);border-color:var(--risk);background:var(--risk-soft)}
h2{font:800 24px/1 var(--display);text-transform:uppercase;letter-spacing:.03em;margin:36px 0 14px}
.tix{display:flex;flex-direction:column;gap:14px}
.ticket{display:grid;grid-template-columns:minmax(0,1fr) 200px;background:var(--surface);border:1px solid var(--line);border-radius:16px;overflow:hidden}
.ticket.hero{border-color:var(--price)}
.main{padding:16px 18px;min-width:0}
.top{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.num{font:800 22px/1 var(--display);color:var(--faint)}
.pname{font:800 26px/1 var(--display);text-transform:uppercase}
.team{font:700 12px/1 var(--display);letter-spacing:.06em;background:var(--surface-2);border:1px solid var(--line);border-radius:6px;padding:4px 7px;color:var(--muted)}
.tier{font:700 11px/1 var(--display);letter-spacing:.12em;text-transform:uppercase;padding:5px 8px;border-radius:6px}
.tier.core{background:var(--edge-soft);color:var(--edge)}.tier.small{background:var(--sel-soft);color:var(--sel)}
.bet{font:800 30px/1.05 var(--display);text-transform:uppercase;margin-top:10px}
.bet small{font:600 14px var(--body);text-transform:none;color:var(--muted);margin-left:6px}
.trend{font:500 13px/1.4 var(--mono);color:var(--muted);margin-top:8px}
.risks{margin:10px 0 0;padding:0;list-style:none;display:flex;flex-direction:column;gap:4px}
.risks li{font-size:13px;color:var(--risk)}.risks li::before{content:"⚠ "}
.ours{margin-top:10px;font-size:13px;color:var(--sel);border-top:1px dashed var(--line);padding-top:8px}
.stub{background:var(--surface-2);border-left:2px dashed var(--line);padding:16px;display:flex;flex-direction:column;justify-content:center;gap:8px;text-align:center}
.price{font:800 44px/.9 var(--display);color:var(--price)}.lbl{font:600 11px/1 var(--body);letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.edge{font:700 14px var(--mono);color:var(--edge)}.edge.neg{color:var(--neg)}
.min{font:600 13px var(--mono);color:var(--fg)}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
table{width:100%;border-collapse:collapse;font-size:13.5px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}
th{font:700 12px/1 var(--display);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.scroll{overflow-x:auto}
.foot{margin-top:44px;color:var(--faint);font-size:12.5px;border-top:1px solid var(--line);padding-top:12px}
@media (max-width:680px){.ticket{grid-template-columns:1fr}.stub{border-left:0;border-top:2px dashed var(--line);flex-direction:row;justify-content:space-between;text-align:left}}
"""


def render(day, tipped, c, spr):
    o = latest(load_cached("live"))
    lo = datetime(day.year, day.month, day.day, 6, tzinfo=ET)
    o = o.filter(pl.col("commence_time").map_elements(lambda s: lo <= datetime.fromisoformat(s) < lo + timedelta(days=1), return_dtype=pl.Boolean))
    tip = {}
    from nba_ladders.market import TEAM_ABBR

    for ht, at, ct in o.select("home_team", "away_team", "commence_time").unique().iter_rows():
        t = datetime.fromisoformat(ct).astimezone(AEDT).strftime("%a %-I:%M%p").replace("AM", "am").replace("PM", "pm")
        tip[TEAM_ABBR.get(ht)] = tip[TEAM_ABBR.get(at)] = t
    snap = o["snapshot"].max() if o.height else "none"
    units = sum(0.25 if b["tier"] == "core" else 0.1 for b in tipped)
    H = [(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>Alt Ladders {day}</title><link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">'
         f'<style>{CSS}</style></head><body><main>'),
         f'<div class="eyebrow">Friend\'s model · tonight\'s card</div><h1>{(day + timedelta(days=1)):%a %-d %b} · 5 tips</h1>',
         ('<p class="lede">Trend-first (rising potentials / rebound chances / cold high-volume shooters against soft matchups), '
         'then a price check against Sportsbet: audited = 0.35 × model + 0.65 × market, bet only at the minimum price or better, '
         'gap over 2× = suspect. Core 0.25u, small 0.1u, max 2 per game, one market per player.</p>'),
         f'<div class="pills"><span class="pill">{len(tipped)} tips · {units:.2f}u</span><span class="pill">Sportsbet prices · snapshot {esc(str(snap))}</span>'
         f'<span class="pill">{len(c["res"])} players modelled · {len(c["cand"])} trend candidates</span>'
         + "".join(f'<span class="pill warn">{esc(t)} spread {s:+.1f}: watch only</span>' for t, s in spr.items() if 9.5 <= abs(s) < 15 and s < 0)
         + '</div><h2>The card</h2><div class="tix">']
    for i, b in enumerate(tipped, 1):
        if b["kind"] == "sgm2":
            bet = f'{esc(b["name"])} {b["ka"]}+ ast &amp; {esc(b["mate"])} {b["ks"]}+ threes'
            name, sub = f'{esc(b["name"])} + {esc(b["mate"].split()[-1])}', "two-player SGM"
            trend = f'legs ${b["pa"]:.2f} × ${b["ps"]:.2f} · joint {b["j"]:.0%} · builder est ${b["price"]:.2f}'
            plabel, edge = "Builder est.", b["e"]
        else:
            bet, name, sub = f'{b["k"]}+ {LAB[b["stat"]]}', esc(b["name"]), f'mean {b["mu"]:.1f} · model {b["q"]:.0%} · audited {b["a"]:.0%}'
            trend = esc(b.get("trend") or "price screen (no trend flag)")
            plabel, edge = "Sportsbet", b["e"]
        H.append(f'<article class="ticket{" hero" if i == 1 else ""}"><div class="main"><div class="top"><span class="num">{i}</span>'
                 f'<span class="pname">{name}</span><span class="team">{esc(b["t"])} v {esc(b["o"])} · {esc(tip.get(b["t"], ""))}</span>'
                 f'<span class="tier {b["tier"]}">{b["tier"]} · {"0.25u" if b["tier"] == "core" else "0.1u"}</span></div>'
                 f'<div class="bet">{bet}<small>{esc(sub)}</small></div><div class="trend">{trend}</div>'
                 f'<ul class="risks">{"".join(f"<li>{esc(x)}</li>" for x in risks(b, c["res"], spr))}</ul>'
                 f'<div class="ours">Our view: {esc(our_view(b))}</div></div>'
                 f'<div class="stub"><div><div class="lbl">{plabel}</div><div class="price">${b["price"]:.2f}</div></div>'
                 f'<div><div class="edge{" neg" if edge < 0 else ""}">{edge:+.0%} edge</div><div class="min">min ${b["min_price"]:.2f}</div></div></div></article>')
    H.append("</div>")
    if c["watch"]:
        H.append('<h2>Watch list (not bet)</h2><div class="panel scroll"><table><tr><th>Player</th><th>Rung</th><th>Price</th><th>Min</th><th>Edge</th><th>Why not</th></tr>')
        for b in c["watch"][:10]:
            H.append(f'<tr><td><b>{esc(b["name"])}</b> <span class="team">{esc(b["t"])}</span></td><td>{b["k"]}+ {LAB[b["stat"]]}</td>'
                     f'<td>${b["price"]:.2f}</td><td>${b["min_price"]:.2f}</td><td>{b["e"]:+.0%}</td><td>{esc(b["why_not"])}</td></tr>')
        H.append("</table></div>")
    H.append('<p class="foot">Port of the friend\'s handoff model (Python; data via nba_api, the official injury report and The Odds API '
             'instead of browser scraping). Re-check prices and injuries before betting: Sportsbet reprices fast on lineup news. '
             'Analysis, not financial advice · Gambling Help 1800 858 858 · BetStop betstop.gov.au</p></main></body></html>')
    return "\n".join(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    a = ap.parse_args()
    tipped, c, spr = tips(a.date)
    load_ours(a.date)
    out = config.ROOT / "docs" / "slates" / f"{a.date}-friend.html"
    out.write_text(render(a.date, tipped, c, spr))
    for i, b in enumerate(tipped, 1):
        if b["kind"] == "sgm2":
            print(f"{i}. [{b['tier']}] {b['name']} {b['ka']}+ ast & {b['mate']} {b['ks']}+ threes · est ${b['price']:.2f} edge {b['e']:+.0%} min ${b['min_price']:.2f}")
        else:
            print(f"{i}. [{b['tier']}] {b['name']} {b['k']}+ {LAB[b['stat']]} @ ${b['price']:.2f} edge {b['e']:+.0%} min ${b['min_price']:.2f} | {our_view(b)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
