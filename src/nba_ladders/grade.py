"""Grade a slate against results: every priced rung (model vs book vs actual) and the bets table.

    uv run python -m nba_ladders.grade --date 2026-10-31   # after pulling that day's game logs
"""
import argparse
import re
from datetime import date

import polars as pl

from nba_ladders import config

ap = argparse.ArgumentParser()
ap.add_argument("--date", type=date.fromisoformat, required=True)
args = ap.parse_args()
md = (config.ROOT / "docs" / "slates" / f"{args.date.isoformat()}.md").read_text().splitlines()
pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(pl.col("game_date") == args.date)
from nba_ladders.sources.injury_report import player_key

act = {player_key(r["player_name"]): r for r in pg.iter_rows(named=True)}
def actual(name, stat):
    r = act.get(player_key(name))
    if r is None: return None
    return sum(r[s] for s in stat.split("+"))
rows, cur = [], None
for ln in md:
    m = re.match(r"### (.+) \(([A-Z]{3})\)$", ln)
    if m: cur = m.group(1); continue
    m = re.match(r"\| (\d+)\+ ([a-z+]+) \| (\$[\d.]+|n/a) \| \$([\d.]+) \(([a-z_]+)\) \| \*\*\$([\d.]+)\*\* \| ([+-−]?\d+)% \| ([^|]*)\|", ln)
    if m and cur:
        k, stat, sb, best, book, fair, ev, note = m.groups()
        a = actual(cur, stat)
        rows.append({"player": cur, "stat": stat, "k": int(k), "best": float(best), "book": book, "fair": float(fair),
                     "ev": float(ev.replace("−", "-")) / 100, "note": note.strip(), "actual": a,
                     "hit": None if a is None else a >= int(k)})
df = pl.DataFrame(rows).unique(["player", "stat", "k"])
played = df.filter(pl.col("actual").is_not_null())
print(f"priced rungs {df.height}, player played {played.height}, DNP {df.height - played.height}")
def summ(lab, d):
    if d.is_empty(): print(lab, "none"); return
    pm, pb = (1 / d["fair"]).sum(), (1 / d["best"]).sum()
    ret = (d["hit"].cast(int) * d["best"]).sum() - d.height
    print(f"{lab:46s} n={d.height:4d} hits {d['hit'].sum():3d} | model expected {pm:6.1f} | book-implied {pb:6.1f} | flat-1u P&L {ret:+7.1f} ({ret/d.height:+.0%})")
summ("all priced rungs", played)
summ("price $3-$10", played.filter(pl.col("best").is_between(3, 10)))
summ("value ($3-10, EV>=10%)", played.filter(pl.col("best").is_between(3, 10), pl.col("ev") >= 0.10))
summ("value, EV 10-50%", played.filter(pl.col("best").is_between(3, 10), pl.col("ev").is_between(0.10, 0.5)))
summ("value, EV >50%", played.filter(pl.col("best").is_between(3, 10), pl.col("ev") > 0.5))
summ("model says negative EV, $3-10", played.filter(pl.col("best").is_between(3, 10), pl.col("ev") < 0))
for st in ("ast", "reb", "pts", "pts+reb", "reb+ast", "pts+ast"):
    summ(f"  value $3-10, {st}", played.filter(pl.col("best").is_between(3, 10), pl.col("ev") >= 0.10, pl.col("stat") == st))
# bets table (best rung per second-tier player)
bt, on = [], False
for ln in md:
    if ln.startswith("## Bets"): on = True; continue
    if on and ln.startswith("## "): break
    m = re.match(r"\| (\d+) \| \*\*(.+?)\*\* \(([A-Z]{3}) vs ([A-Z]{3})\) \|.*?\| \*\*(\d+)\+ ([a-z+]+) \$([\d.]+)\*\*", ln) if on else None
    if m:
        _, n, t, o, k, st, pr = m.groups(); a = actual(n, st)
        bt.append({"player": n, "bet": f"{k}+ {st}", "price": float(pr), "actual": a, "hit": None if a is None else a >= int(k)})
b = pl.DataFrame(bt)
with pl.Config(tbl_rows=20, tbl_hide_column_data_types=True):
    print(b)
pl_ = b.filter(pl.col("hit").is_not_null())
print("bets table: hits", pl_["hit"].sum(), "/", pl_.height, "flat P&L", round((pl_["hit"].cast(int) * pl_["price"]).sum() - pl_.height, 2))
