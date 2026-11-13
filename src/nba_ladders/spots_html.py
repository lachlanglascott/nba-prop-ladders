"""Standalone HTML for the spots slate: grade badges, situation chips, coloured matchups,
projection panel, confidence bars. Light/dark, mobile-friendly, no external assets."""

from html import escape as esc

CSS = """
:root{--bg:#f4f5f2;--card:#fff;--ink:#1b1e23;--muted:#646b76;--line:#e1e4e8;--soft:#f6f7f9;
--a:#18794e;--a-bg:#dff3e8;--b:#2457c5;--b-bg:#e3ebfb;--c:#5d6470;--c-bg:#eceef1;
--role:#6a3fc1;--role-bg:#efe8fb;--bounce:#a05a00;--bounce-bg:#fcefd9;--opp:#0f7b80;--opp-bg:#ddf3f3;
--warn:#a33a1f;--warn-bg:#fbe7e1;--good:#18794e;--bad:#b42318;--accent:#2457c5}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121418;--card:#1c1f25;--ink:#e8eaed;--muted:#9aa2ad;
--line:#2d323a;--soft:#232730;--a:#5fd39a;--a-bg:#17322a;--b:#8fb0ff;--b-bg:#1d2a45;--c:#a7aeb8;--c-bg:#2a2f37;
--role:#b99bff;--role-bg:#2c2440;--bounce:#f3b860;--bounce-bg:#3a2c16;--opp:#6fd6da;--opp-bg:#163536;--warn:#ff9a7f;
--warn-bg:#3c211b;--good:#5fd39a;--bad:#ff8a80;--accent:#8fb0ff}}
:root[data-theme=dark]{--bg:#121418;--card:#1c1f25;--ink:#e8eaed;--muted:#9aa2ad;--line:#2d323a;--soft:#232730;--a:#5fd39a;
--a-bg:#17322a;--b:#8fb0ff;--b-bg:#1d2a45;--c:#a7aeb8;--c-bg:#2a2f37;--role:#b99bff;--role-bg:#2c2440;--bounce:#f3b860;
--bounce-bg:#3a2c16;--opp:#6fd6da;--opp-bg:#163536;--warn:#ff9a7f;--warn-bg:#3c211b;--good:#5fd39a;--bad:#ff8a80;--accent:#8fb0ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:26px 16px 64px}
h1{font-size:26px;margin:0;letter-spacing:-.01em}h2{font-size:18px;margin:34px 0 12px}h3{margin:0;font-size:17px}
.sub{color:var(--muted);margin:4px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 0}
.chip{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:2px 10px;font-size:12.5px;font-weight:600;background:var(--soft);color:var(--muted);white-space:nowrap}
.chip.role{background:var(--role-bg);color:var(--role)}.chip.bounce{background:var(--bounce-bg);color:var(--bounce)}
.chip.opp{background:var(--opp-bg);color:var(--opp)}.chip.warn{background:var(--warn-bg);color:var(--warn)}
.chip.good{background:var(--a-bg);color:var(--a)}
.badge{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:8px;font-weight:800;font-size:15px}
.badge.A{background:var(--a-bg);color:var(--a)}.badge.B{background:var(--b-bg);color:var(--b)}.badge.C{background:var(--c-bg);color:var(--c)}
details.how{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin-top:18px}
details.how summary{cursor:pointer;font-weight:600}details.how p{color:var(--muted);margin:8px 0 0}
.card-list{display:flex;flex-direction:column;gap:10px}
.row{display:grid;grid-template-columns:44px minmax(0,1.3fr) minmax(0,1.5fr) minmax(0,1fr);gap:14px;align-items:center;
background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.row .who b{font-size:15.5px}.row .who small{display:block;color:var(--muted)}
.bet{font-weight:700;font-size:16px}.bet small{font-weight:500;color:var(--muted);font-size:13px}
.nums{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13.5px;color:var(--muted);margin-top:3px}
.nums b{color:var(--ink)}.take{color:var(--accent);font-weight:800}
.ev{font-weight:800}.ev.pos{color:var(--good)}.ev.neg{color:var(--bad)}
.sgm{font-size:13px;color:var(--muted);margin-top:6px;border-top:1px dashed var(--line);padding-top:6px}
.why{font-size:13.5px;color:var(--ink)}
ul.also{margin:4px 0 0;padding-left:16px}ul.also li{margin:2px 0}
.projline{font-size:13px;margin-top:5px;padding:4px 8px;border-radius:7px;background:var(--soft);color:var(--ink);font-weight:600}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.panel.warn{border-color:var(--warn);background:var(--warn-bg)}
.cond{display:grid;gap:10px}.cond .panel h3{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
ul{margin:8px 0 0;padding-left:18px}li{margin:3px 0}
.spot{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0}
.spot-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:flex-start}
.conf{min-width:170px}.conf .lbl{font-size:12px;color:var(--muted)}.bar{height:7px;border-radius:5px;background:var(--soft);overflow:hidden;margin-top:4px}
.bar i{display:block;height:100%;background:var(--accent)}
.proj{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:14px 0 6px}
.stat{background:var(--soft);border-radius:10px;padding:8px 10px}.stat .v{font-size:20px;font-weight:800}.stat .l{font-size:12px;color:var(--muted)}
.note{font-size:13px;color:var(--muted);margin:6px 0 0}
.scroll{overflow-x:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-size:12px;color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums}
.games td{color:var(--muted)}.games td b{color:var(--ink)}
.foot{color:var(--muted);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px}
@media (max-width:760px){.row{grid-template-columns:40px minmax(0,1fr)}.row>.betcol,.row>.why{grid-column:1/-1}.proj{grid-template-columns:repeat(2,1fr)}}
"""

KIND = {"role": ("role", "Role change"), "bounce": ("bounce", "Bounce"), "emerging": ("opp", "Volume up")}
STAKE = {"A": "1u", "B": "0.5u", "C": "0.25u"}


def _ev(ev):
    return f'<span class="ev {"pos" if ev > 0 else "neg"}">{ev:+.0%}</span>'


def _bet_html(b, big=True):
    est = "est. " if b["kind"] == "SGM" else ""
    book = b["book"].replace("_au", "").replace("sportsbet", "Sportsbet").replace("betr", "Betr")
    label = ("SGM " if b["kind"] == "SGM" else "") + b["bet"]
    out = (f'<div class="bet">{esc(label)} <small>{book} · {est}${b["price"]:.2f}</small></div>' if big
           else f'<b>{esc(label)}</b> {book} {est}${b["price"]:.2f}')
    nums = (f'<div class="nums"><span>fair <b>${1 / b["p"]:.2f}</b></span><span>EV {_ev(b["ev"])}</span>'
            f'<span>take at <span class="take">${b["take_at"]:.2f}</span></span>'
            + (f'<span>{esc(b["hist"])}</span>' if b.get("hist") else "") + '</div>')
    return out + (nums if big else f' · fair ${1 / b["p"]:.2f} · EV {_ev(b["ev"])} · take at <span class="take">${b["take_at"]:.2f}</span>')


def _opp_chips(a):
    f = a.get("opp_f") or {}
    if not f:
        return ""
    name = {"G": "guards", "W": "wings", "B": "bigs"}
    grp = a.get("opp_line", "").split(" vs ")[1].split(":")[0] if " vs " in a.get("opp_line", "") else ""
    chips = []
    for s, lab in (("pts", "pts"), ("reb", "reb"), ("ast", "ast")):
        v = (f.get(s, 1) - 1) * 100
        cls = "good" if v >= 2 else "warn" if v <= -2 else ""
        chips.append(f'<span class="chip {cls}">{lab} {v:+.0f}%</span>')
    return f'<span class="chip opp">{esc(a["opp"])} vs {esc(grp or name.get("W"))}</span>' + "".join(chips)


def _proj(a, b):
    from nba_ladders.spots import also_bets, kelly, proj_line

    s = proj_line(a, b)
    xs = also_bets(a, b)
    also = ""
    if xs:
        also = ('<div class="sgm"><b>His other stats</b> (best rung each, fair or better):<ul class="also">'
                + "".join(f'<li><b>{esc(x["bet"])}</b> {x["book"].replace("_au", "")} ${x["price"]:.2f} · {x["p"]:.0%} chance · '
                          f'fair ${1 / x["p"]:.2f} · EV {_ev(x["ev"])} · Kelly {kelly(x):.1%}'
                          + (' <span class="chip">≈ fair, skip</span>' if x["ev"] < 0.04 else "") + "</li>" for x in xs) + "</ul></div>")
    ra = sorted((r for r in (a.get("ra_ladder") or []) + (a.get("pr_ladder") or []) if r["ev"] is not None and r["ev"] >= 0.04),
                key=lambda r: -r["ev"])[:3]
    if ra:
        also += ('<div class="sgm"><b>Combos</b> (SGM, check the builder): '
                 + " · ".join(f'<b>{esc(r["bet"])}</b> est ${r["est"]:.2f}, fair ${r["fair"]:.2f}, bet at <span class="take">${r["bet_at"]:.2f}</span>'
                              f' (strong leg: {esc(r["strong"])} {r["strong_p"]:.0%})'
                              for r in ra) + "</div>")
    return (f'<div class="projline">{esc(s)} · Kelly {kelly(b):.1%}</div>' if s else "") + also


def _why(a, b):
    from nba_ladders.spots import _why as why

    return why(a, b)


def _spot_html(a: dict) -> list[str]:
    """One spot card: situation, projection, last 5, matching games, opponent comps, bets."""
    H = []
    s = a["sim_means"]
    chips = []
    for x in a["sits"]:
        k = KIND.get(x["kind"], ("", x["kind"]))
        chips.append(f'<span class="chip {k[0]}">{k[1]}</span>')
    if a.get("own_p", 0) >= 0.2:
        chips.append(f'<span class="chip warn">His own P(sits) {a["own_p"]:.0%} · void if out</span>')
    for n, p in a.get("unc", {}).items():
        chips.append(f'<span class="chip warn">{esc(n)} P(sits) {p:.0%}</span>')
    H.append(f'<div class="spot"><div class="spot-head"><div><h3>{esc(a["name"])} <span class="chip">{esc(a["team"])} vs {esc(a["opp"])}</span>'
             + (f' <span class="chip">spread {a["spread"]:+.1f}</span>' if a["spread"] is not None else "")
             + f'</h3><div class="chips">{"".join(chips)}</div></div>'
             f'<div class="conf"><div class="lbl">confidence {a["conf"]:.2f} (model weight vs market)</div>'
             f'<div class="bar"><i style="width:{min(a["conf"] / 0.35, 1) * 100:.0f}%"></i></div></div></div><ul>')
    for x in a["sits"]:
        H.append(f"<li>{esc(x['text'])}</li>")
    H.append(f'</ul><div class="chips">{_opp_chips(a)}</div>')
    lo, hi = a["min_p10_90"]
    H.append('<div class="proj">'
             f'<div class="stat"><div class="v">{s["minutes"]:.0f}</div><div class="l">minutes ({lo:.0f}–{hi:.0f})</div></div>'
             f'<div class="stat"><div class="v">{s["pts"]:.1f}</div><div class="l">points</div></div>'
             f'<div class="stat"><div class="v">{s["reb"]:.1f}</div><div class="l">rebounds · {a["rc36"]:.1f} chances/36 at {a["cr"]:.0%}</div></div>'
             f'<div class="stat"><div class="v">{s["ast"]:.1f}</div><div class="l">assists · {a["pot36"]:.1f} potential/36 at {a["ca"]:.0%}</div></div></div>'
             f'<p class="note">Projection from {esc(a["why_sample"] or "his baseline")}.</p>')
    def games_table(df, title):
        also = "also_out" in df.columns
        rows = df.select("game_date", "opponent", "minutes", "pts", "reb", "ast", "potential_ast", "reb_chances").rows()
        extra = df["also_out"].to_list() if also else [""] * df.height
        H.append(f'<p class="note"><b>{title}</b></p><div class="scroll"><table class="games"><tr><th>Game</th><th>Min</th><th>Pts</th>'
                 '<th>Reb</th><th>Ast</th><th>Pot ast</th><th>Reb ch</th>' + ("<th>Also out then (plays tonight)</th>" if also else "") + "</tr>")
        for (d, o, mi, p_, r, x, pa, rc), e in list(zip(rows, extra))[::-1]:
            H.append(f'<tr><td>{d:%d %b} vs {esc(o)}</td><td class="num"><b>{mi}</b></td><td class="num">{p_}</td><td class="num">{r}</td>'
                     f'<td class="num">{x}</td><td class="num">{"" if pa is None else pa}</td><td class="num">{"" if rc is None else rc}</td>'
                     + (("<td>" + (f'<span class="chip warn">{esc(e)} · counts less</span>' if e else "same lineup") + "</td>") if also else "") + "</tr>")
        H.append("</table></div>")

    if a.get("last5") is not None and a["last5"].height:
        games_table(a["last5"], "Last 5 games (any lineup)")
    if a["sample"] is not None and a["sample"].height:
        from nba_ladders.spots import SEASON

        this = a["sample"].filter(a["sample"]["season_id"] == SEASON)
        n_last = a["sample"].height - this.height
        extra = f" · plus {n_last} last-season games used in the projection, not shown" if n_last else ""
        if this.height:
            games_table(this.tail(6), f"Games that match tonight, this season ({esc(a['why_sample'].split(' (')[0])}){extra}")
        elif n_last:
            H.append(f'<p class="note">No game this season matches tonight\'s lineup; the projection uses {n_last} last-season games.</p>')
    for key, title, why in (("ra_ladder", "Rebounds + assists ladder (SGM)", ("he does both, so R and A rise together. "
                             "Builders pay about independent odds on <b>uneven</b> combos (one easy leg, one hard), so those are where the value is.")),
                            ("pr_ladder", "Points + rebounds ladder (SGM)", ("for bigs, minutes drive both, so P and R land together "
                             "(12p/10r 1.36x independent when minutes swing). Builders already charge for that link (about 0.78x the legs), "
                             "so the value comes from the spot, not the builder."))):
        if not a.get(key):
            continue
        H.append(f'<p class="note"><b>{title}</b>: {why} Take a combo only at the <span class="take">bet at</span> builder price or better.</p>'
                 '<div class="scroll"><table class="games"><tr><th>Combo</th><th>Our chance</th><th>Fair</th><th>Bet at</th><th>Builder est.</th><th>EV at est.</th><th>History</th></tr>')
        for r in a[key]:
            est = f'${r["est"]:.2f} ({r["book"].replace("_au", "")})' if r["est"] else "—"
            ev = _ev(r["ev"]) if r["ev"] is not None else ""
            H.append(f'<tr><td><b>{esc(r["bet"])}</b></td><td class="num">{r["p"]:.0%}</td><td class="num">${r["fair"]:.2f}</td>'
                     f'<td class="num"><span class="take">${r["bet_at"]:.2f}+</span></td><td class="num">{est}</td><td class="num">{ev}</td>'
                     f'<td class="note">{esc(r["hist"])}</td></tr>')
        H.append("</table></div>")
    oc = a.get("opp_comps") or {}
    if oc.get("rows"):
        dd = oc["diff"]
        col = lambda v: "good" if v >= 0.5 else "warn" if v <= -0.5 else ""
        H.append(f'<p class="note"><b>{esc(oc["label"].capitalize())} vs {esc(a["opp"])}, its last 5 games</b> (vs their own averages): '
                 + " ".join(f'<span class="chip {col(dd[s])}">{lab} {dd[s]:+.1f}</span>' for s, lab in (("pts", "pts"), ("reb", "reb"), ("ast", "ast")))
                 + ' <span class="note">· a handful of games is mostly noise; the opponent factor above is the tested number</span></p>')
        H.append('<div class="scroll"><table class="games"><tr><th>Game</th><th>Player</th><th>Min</th><th>Pts</th><th>Reb</th><th>Ast</th></tr>')
        for d, tm, nm, mi, p_, r, x, dp, dr, dx in oc["rows"]:
            sg = lambda v: f' <span class="note">({v:+.0f})</span>'
            H.append(f'<tr><td>{d:%d %b} ({esc(tm)})</td><td><b>{esc(nm)}</b></td><td class="num">{mi}</td>'
                     f'<td class="num">{p_}{sg(dp)}</td><td class="num">{r}{sg(dr)}</td><td class="num">{x}{sg(dx)}</td></tr>')
        H.append("</table></div>")
    H.append('<div class="scroll"><table><tr><th></th><th>Bet</th><th>Book</th><th>Price</th><th>Model</th><th>Market</th>'
             '<th>Honest fair</th><th>EV</th><th>Take at</th></tr>')
    for b in a["bets"]:
        book = b["book"].replace("_au", "")
        flag = ' <span class="chip warn">check live</span>' if b.get("outlier") else (' <span class="chip">SGM version</span>' if b.get("alt") else "")
        H.append(f'<tr><td><span class="badge {b["grade"]}">{b["grade"]}</span></td><td><b>{"SGM " if b["kind"] == "SGM" else ""}{esc(b["bet"])}</b>{flag}<br><span class="note">{esc(b.get("hist", ""))}</span></td>'
                 f'<td>{book}</td><td class="num">{"est. " if b["kind"] == "SGM" else ""}${b["price"]:.2f}</td>'
                 f'<td class="num">${1 / b["p_model"]:.2f}</td><td class="num">${1 / b["p_mkt"]:.2f}</td>'
                 f'<td class="num"><b>${1 / b["p"]:.2f}</b></td><td class="num">{_ev(b["ev"])}</td>'
                 f'<td class="num"><span class="take">${b["take_at"]:.2f}</span></td></tr>')
    H.append("</table></div></div>")
    return H


def render_html(res: dict, cond: list[dict]) -> str:
    day = res["day"]
    unfiled = sorted(res["teams"] - res["filed"])
    head = [f'<span class="chip">Injury report {esc(res["report_time"])} ET</span>',
            f'<span class="chip">Odds {esc(str(res["snapshot"]))}</span>',
            f'<span class="chip">{res["looked"]} players checked · {len(res["spots"])} spots</span>']
    if unfiled:
        head.append(f'<span class="chip warn">Not yet filed: {esc(", ".join(unfiled))}</span>')
    rec = res.get("record")
    if rec:
        cls = "good" if rec["pnl"] >= 0 else "warn"
        head.append(f'<span class="chip {cls}">Live record ({rec["days"]} slate{"s" * (rec["days"] > 1)}, card + triggered ifs): '
                    f'{rec["hits"]}/{rec["n"]} (expected {rec["exp"]:.1f}) · {rec["pnl"]:+.1f}u at 1u</span>')
    rp = res.get("replay")
    if rp:
        head.append(f'<span class="chip">Replay (in-sample, {rp["days"]} slates): {rp["hits"]}/{rp["n"]} · {rp["pnl"]:+.1f}u</span>')
    H = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
         f'<title>Spots {day}</title><style>{CSS}</style></head><body><main>',
         f'<h1>Spots · {day:%a %d %b}</h1><p class="sub">US Eastern slate · situation-first · honest prices</p>',
         f'<div class="chips">{"".join(head)}</div>',
         ('<details class="how"><summary>How to read this</summary><p>A spot needs a reason the player is different tonight: '
         'a <b>role change</b> from tonight\'s lineup (his own games without those teammates, this season and last) or a '
         '<b>bounce</b> (normal volume, conversion well below his career rate). Opposition is shown on every spot and is in '
         'the projection. <b>Fair and EV are honest</b>: they start from the market\'s real probability (alt overs hit '
         '~93% of implied at $1.25, ~77% at $9) and move toward the model only by the confidence bar. SGM prices are '
         'builder estimates (0.9× the AU-median legs): take only at the <span class="take">take at</span> price or better. '
         'Grades: A = lineup role spot with strong evidence; B = solid; C = thin.</p></details>')]
    # the card
    from nba_ladders.spots import card_bets

    card = card_bets(res, cond=cond)
    H.append('<h2>The card: 5 tips</h2>')
    firm = sum(not b.get("if_sits") and not b.get("lean") for _, b in card)
    if firm < 3:
        H.append(f'<p class="note">Only {firm} firm tip{"s" * (firm != 1)} clear the bar tonight. The next best are in '
                 f'<b>{day.isoformat()}-next-best.html</b> (same folder).</p>')
    H.append('<div class="card-list">')
    for a, b in card:
        sg = next((x for x in a["bets"] if x["kind"] == "SGM" and x is not b), None)
        kinds = "".join(f'<span class="chip {KIND[x["kind"]][0]}">{KIND[x["kind"]][1]}</span>' for x in a["sits"] if x["kind"] in KIND)
        g, stake = ("C", "0.25u · lean") if b.get("lean") else ("B", "0.5u · only if he sits") if b.get("if_sits") else (b["grade"], STAKE[b["grade"]])
        mark = "L" if b.get("lean") else "IF" if b.get("if_sits") else g
        cond_chip = f'<span class="chip warn">Only if {esc(b["if_sits"])}</span>' if b.get("if_sits") else ""
        H.append(f'<div class="row"><span class="badge {g}">{mark}</span>'
                 f'<div class="who"><b>{esc(a["name"])}</b><small>{esc(a["team"])} vs {esc(a["opp"])} · stake {stake}</small>'
                 f'<div class="chips">{cond_chip}{kinds}</div></div>'
                 f'<div class="betcol">{_bet_html(b)}{_proj(a, b)}' + (f'<div class="sgm">SGM version: {_bet_html(sg, big=False)}</div>' if sg else "")
                 + f'</div><div class="why">{esc(_why(a, b))}<div class="chips">{_opp_chips(a)}</div></div></div>')
    H.append("</div>")
    from nba_ladders.spots import price_plays

    pp = price_plays(res, card)
    if pp:
        H.append('<h2>Price plays: top 5</h2><p class="note">No lineup angle: our projection (minutes with tonight\'s blowout risk) '
                 'against the price. A step below the card: <b>0.25u</b>, at the <span class="take">take at</span> price or better.</p>'
                 '<div class="card-list">')
        for i, (a, b) in enumerate(pp, 1):
            H.append(f'<div class="row"><span class="badge C">{i}</span><div class="who"><b>{esc(a["name"])}</b>'
                     f'<small>{esc(a["team"])} vs {esc(a["opp"])} · stake 0.25u</small><div class="chips"><span class="chip">No lineup angle</span></div></div>'
                     f'<div class="betcol">{_bet_html(b)}{_proj(a, b)}</div><div class="why">{esc(_why(a, b))}<div class="chips">{_opp_chips(a)}</div></div></div>')
        H.append("</div>")
    ft = res.get("friend") or {}
    if ft.get("bets") or ft.get("ideas"):
        from nba_ladders.friend_tips import idea_note
        from nba_ladders.friend_tips import note as ft_note

        H.append('<h2>Friend\'s tips: assist trends</h2><p class="note">Their trend-first model, kept where it backtests: assist '
                 'trends with <b>minutes flat</b> (+8% at Sportsbet 2024-26, CI −3 to +19%). Rebound trends dropped (−21%). Honest fair = '
                 'market\'s real chance × the trigger\'s backtested lift. <b>0.25u</b>, at the <span class="take">take at</span> price or better.</p>'
                 '<div class="card-list">')
        for b in ft.get("bets", []):
            tag = "Bet" if b["bet_ok"] else "Watch"
            H.append(f'<div class="row"><span class="badge {"C" if b["bet_ok"] else ""}">{tag}</span><div class="who"><b>{esc(b["name"])}</b>'
                     f'<small>{esc(b["team"])} vs {esc(b["opp"])}</small></div><div class="betcol"><b>{esc(b["bet"])}</b> (sportsbet) ${b["price"]:.2f} · '
                     f'fair ${1 / b["p"]:.2f} · EV {b["ev"]:+.0%} · take at <span class="take">${b["take_at"]:.2f}</span>'
                     f'<div class="note">their model {b["p_model"]:.0%} (exp {b["mu"]:.1f})</div></div>'
                     f'<div class="why">{esc(b["why"])}<div class="note">{esc(ft_note(b))}</div></div></div>')
        H.append("</div>")
        if ft.get("ideas"):
            H.append('<h3>Ideas: their rebound trends and threes</h3><p class="note">Trends plus a threes price screen (their model 1.15-2× the market). Not bets: rebound trends lost 21% in the backtest, '
                     'threes are untested. Shown as leads and graded so we learn. Fair = the market\'s own chance.</p><div class="card-list">')
            for b in ft["ideas"]:
                H.append(f'<div class="row"><span class="badge">Idea</span><div class="who"><b>{esc(b["name"])}</b>'
                         f'<small>{esc(b["team"])} vs {esc(b["opp"])}</small></div><div class="betcol"><b>{esc(b["bet"])}</b> (sportsbet) ${b["price"]:.2f} · '
                         f'market fair ${1 / b["p_mkt"]:.2f}<div class="note">their model {b["p_model"]:.0%} (exp {b["mu"]:.1f})</div></div>'
                         f'<div class="why">{esc(b["why"])}<div class="note">{esc(idea_note(b))}</div></div></div>')
            H.append("</div>")
    passed = [(a, b) for a in res["spots"] + res.get("leans", []) for b in a["bets"][:1] if b.get("h") == "gate"]
    if passed:
        H.append('<h2>Passed: bad opposition spot</h2><p class="note">Angle H (docs/opp_position.md): a big\'s points / rebounds '
                 'bet where the opponent allows less than 1.02 of normal to bigs lost 12-14% at Sportsbet in both seasons. Not bets.</p>'
                 '<div class="panel warn"><ul>')
        for a, b in passed:
            H.append(f'<li><b>{esc(a["name"])}</b> ({esc(a["team"])} vs {esc(a["opp"])}): {_bet_html(b, big=False)} · '
                     f'{esc(b["hist"].split(" · ")[-1])}'
                     + "".join(f" · up against {esc(x)}" for x in a.get("opp_line", "").split(" · ") if " tonight: " in x) + '</li>')
        H.append("</ul></div>")
    wait = res.get("waiting", [])
    if wait:
        H.append('<h2>Spots waiting for prices</h2><p class="note">Real lineup spots in games where the books haven\'t posted '
                 'player props yet. Re-run once they\'re up; until then, a rung is a bet only at the <span class="take">bet at</span> '
                 'price or better (the honest EV test, assuming the book prices like the market usually does).</p><div class="cond">')
        for a in sorted(wait, key=lambda a: -a["conf"]):
            s, (lo, hi) = a["sim_means"], a["min_p10_90"]
            kinds = "".join(f'<span class="chip {KIND[x["kind"]][0]}">{KIND[x["kind"]][1]}</span>' for x in a["sits"] if x["kind"] in KIND)
            H.append(f'<div class="panel"><h3>{esc(a["name"])} <span class="chip">{esc(a["team"])} vs {esc(a["opp"])}</span>{kinds}</h3>'
                     f'<p class="note">{esc("; ".join(x["text"] for x in a["sits"]))}</p>'
                     f'<div class="projline">Exp {s["minutes"]:.0f} min ({lo:.0f}–{hi:.0f}) · {s["pts"]:.1f}p / {s["reb"]:.1f}r / {s["ast"]:.1f}a</div>'
                     '<div class="scroll"><table><tr><th>Rung</th><th>Our chance</th><th>Model fair</th><th>Bet at</th><th>History</th></tr>')
            for r in a["ladder"]:
                H.append(f'<tr><td><b>{esc(r["bet"])}</b></td><td class="num">{r["p_model"]:.0%}</td><td class="num">${r["fair"]:.2f}</td>'
                         f'<td class="num"><span class="take">${r["bet_at"]:.2f}+</span></td><td class="note">{esc(r["hist"])}</td></tr>')
            H.append("</table></div></div>")
        H.append("</div>")
    if cond:
        H.append('<h2>Key questionable players: the bet if they miss</h2><div class="cond">')
        for c in cond:
            H.append(f'<div class="panel"><h3>{esc(c["player"])} <span class="chip">{esc(c["team"])} · {c["mpg"]:.0f} min/g</span>'
                     f'<span class="chip warn">P(sits) {c["p"]:.0%}</span></h3><p class="note">{esc(c["reason"])}</p>')
            if c["bets"]:
                H.append("<ul>")
                for a, b in c["bets"]:
                    H.append(f'<li>If he misses: <b>{esc(a["name"])}</b> {_bet_html(b, big=False)} · <span class="note">{esc(_why(a, b))}</span></li>')
                H.append("</ul>")
            else:
                H.append('<p class="note">No bet clears if he misses: thin evidence for his teammates without him, '
                         'or the market has already moved.</p>')
            H.append("</div>")
        H.append("</div>")
    shop = [(a, b) for a in res["spots"] for b in a["bets"] if b.get("outlier")]
    if shop:
        H.append('<h2>Check these prices are live</h2><div class="panel warn"><ul>')
        for a, b in sorted(shop, key=lambda ab: -ab[1]["ev"])[:6]:
            H.append(f'<li><b>{esc(a["name"])}</b> ({esc(a["team"])}): {_bet_html(b, big=False)} · market fair ${1 / b["p_mkt"]:.2f}</li>')
        H.append("</ul></div>")
    # spots
    H.append("<h2>Spots</h2>")
    for a in res["spots"]:
        H += _spot_html(a)
    H.append('<p class="foot">Personal model, public data (NBA.com box scores and tracking, official injury reports, The Odds API). '
             'Gamble responsibly · Gambling Help 1800 858 858 · BetStop betstop.gov.au</p></main></body></html>')
    return "\n".join(H)


def render_next_html(res: dict, card: list, picks: list) -> str:
    """Secondary doc for thin nights: the card's tips, then the next best to make at least 3."""
    from nba_ladders.spots import MIN_TIPS, PRIMARY

    day = res["day"]
    firm = sum(not b.get("if_sits") and not b.get("lean") for _, b in card)
    need = MIN_TIPS - firm
    H = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
         f'<title>Next Best {day}</title><style>{CSS}</style></head><body><main>',
         (f'<h1>Next best · {day:%a %d %b}</h1><p class="sub">Only {firm} firm tip{"s" * (firm != 1)} cleared the card tonight{" (the rest are leans or IF bets that may not trigger)" if firm < len(card) else ""}. '
         f'These are the next {need} to make {MIN_TIPS}, plus reserves in case a price has moved.</p>'),
         ('<div class="panel"><p class="note" style="margin:0">These are a step below card quality. Stake them smaller than the card: '
         '<b>0.25u</b>, or 0.5u for the IF bets once the player is confirmed out. A negative EV means the price is just under our honest fair. '
         'Take those only at the <span class="take">take at</span> price or better. "No lineup angle" means a projection against the price, '
         'without a teammate change behind it.</p></div>'),
         '<h2>On the card</h2><div class="card-list">']
    for a, b in card:
        H.append(f'<div class="row"><span class="badge {b.get("grade", "C")}">{b.get("grade", "C")}</span><div class="who"><b>{esc(a["name"])}</b>'
                 f'<small>{esc(a["team"])} vs {esc(a["opp"])}</small></div><div class="betcol">{_bet_html(b)}{_proj(a, b)}</div>'
                 f'<div class="why">{esc(_why(a, b))}</div></div>')
    H.append("</div>")
    for title, rows in (("Next best", picks[:need]), ("Reserves", picks[need:])):
        if not rows:
            continue
        H.append(f'<h2>{title}</h2><div class="card-list">')
        for i, (a, b) in enumerate(rows, 1):
            angle = any(x["kind"] in PRIMARY for x in a["sits"])
            tag = ('<span class="chip warn">Only if ' + esc(b["if_sits"]) + '</span>' if b.get("if_sits")
                   else '<span class="chip role">Lineup angle</span>' if angle
                   else '<span class="chip opp">Trend: minutes + chances up</span>' if a.get("trend") else '<span class="chip">No lineup angle</span>')
            if b["ev"] < 0:
                tag += '<span class="chip warn">Just under fair: take-at price only</span>'
            H.append(f'<div class="row"><span class="badge C">{i}</span><div class="who"><b>{esc(a["name"])}</b>'
                     f'<small>{esc(a["team"])} vs {esc(a["opp"])} · stake 0.25u</small><div class="chips">{tag}</div></div>'
                     f'<div class="betcol">{_bet_html(b)}{_proj(a, b)}</div><div class="why">{esc(_why(a, b))}<div class="chips">{_opp_chips(a)}</div></div></div>')
        H.append("</div>")
    H.append("<h2>The detail</h2>")
    seen = set()
    for a, _ in picks:
        if a["name"] not in seen and "sim_means" in a:
            seen.add(a["name"])
            H += _spot_html(a)
    H.append('<p class="foot">Personal model, public data (NBA.com box scores and tracking, official injury reports, The Odds API). '
             'Gamble responsibly · Gambling Help 1800 858 858 · BetStop betstop.gov.au</p></main></body></html>')
    return "\n".join(H)
