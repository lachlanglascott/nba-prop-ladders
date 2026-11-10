"""Offline test of how spots blends per-minute volume in a new role (a key teammate out).

    uv run python -m nba_ladders.rate_eval            # 2024-25 + 2025-26

For each game where a key teammate (24+ mpg, usually plays) is out, predict the player's rebound chances,
potential assists and points from his earlier games this season with that teammate out (recency-weighted,
shrunk to his normal-lineup rate), and score the count given his actual minutes. Variants:
  - half-life of the recency weights (games) and pseudo-games of baseline (k)
  - lineup weighting of matching games with extra absences: none / minutes table (current) / per-stat
    de-inflation (divide the game's stat by the measured lift of the extra absences, fit on the other season)
Question (Dosunmu without Edwards, Nov 2026): do recent matching games deserve more weight than half-life 3?
"""

from collections import defaultdict

import numpy as np
import polars as pl

from nba_ladders import config
from nba_ladders.absences import position_groups
from nba_ladders.luck import load as load_tracking
from nba_ladders.scenarios import positions
from nba_ladders.spots import (
    EXTRA_OUT_BY_POS,
    EXTRA_OUT_W,
    KEY_MPG,
    KEY_SHARE,
    LW_FLOOR,
    _game_table,
)

COLS = ("reb_chances", "potential_ast", "pts")
SEASONS = ("22024", "22025")


def _per36(v: np.ndarray, m: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones(len(v)) if w is None else w
    return float(np.sum(v * w) / np.sum(m * w) * 36)


def events(season: str, pg: pl.DataFrame, trk: pl.DataFrame, grp: dict) -> list[dict]:
    """Each new-role game with the player's prior history (matching + normal-lineup games)."""
    gt = _game_table(pg, trk, {}, season=season).drop_nulls(list(COLS[:2]))
    cur = pg.filter(pl.col("season_id") == season)
    mpg = dict(cur.group_by("player_id").agg(pl.col("minutes").mean()).iter_rows())
    out = []
    for (pid,), g in gt.group_by("player_id"):
        g = g.sort("game_date")
        if g.height < 8:
            continue
        ab = [set(a) for a in g["absent"].to_list()]
        abw = [set(a) for a in g["absent_wide"].to_list()]
        mins = g["minutes"].to_numpy().astype(float)
        vals = {c: g[c].to_numpy().astype(float) for c in COLS}
        for i in range(5, g.height):
            if mins[i] < 12:
                continue
            prior = range(i)
            key = {m for m in ab[i] if mpg.get(m, 0) >= KEY_MPG and sum(m not in ab[j] for j in prior) / i >= KEY_SHARE}
            if not key:
                continue
            comp = [j for j in prior if key <= ab[j] and mins[j] > 0]
            base = [j for j in prior if not (key & ab[j]) and mins[j] > 0]
            if len(base) < 3 or not comp:
                continue
            playing = {m for j in prior for m in abw[j]} - abw[i]  # teammates who play tonight
            extra = [sorted((abw[j] & playing) - {pid}) for j in comp]
            out.append({"pid": pid, "season": season, "grp": grp.get(pid, "W"), "i": i, "n": len(comp),
                        "comp": comp, "base": base, "extra": extra, "mins": mins, "vals": vals,
                        "extra_grp": [[grp.get(m, "W") for m in e] for e in extra]})
    return out


def lift_table(evs: list[dict]) -> dict:
    """Measured per-36 lift of each stat in a matching game per extra absence, by (my group, their group):
    matching games with that one extra absence vs matching games with none, same player-event."""
    num, den = defaultdict(float), defaultdict(float)
    for e in evs:
        ix = {j: k for k, j in enumerate(e["comp"])}
        clean = [j for j in e["comp"] if not e["extra"][ix[j]]]
        if not clean:
            continue
        for j in e["comp"]:
            eg = e["extra_grp"][ix[j]]
            if len(eg) != 1:
                continue
            for c in COLS:
                r0 = _per36(e["vals"][c][clean], e["mins"][clean])
                r1 = e["vals"][c][j] / max(e["mins"][j], 1) * 36
                if r0 > 0:
                    key = (c, e["grp"], eg[0])
                    num[key] += np.log(max(r1, 0.5) / r0) * e["mins"][j]
                    den[key] += e["mins"][j]
    # shrink log-lift toward 0 with 300 minutes of prior
    return {k: float(np.exp(num[k] / (den[k] + 300.0))) for k in num}


def predict(e: dict, c: str, half: float, k: float, mode: str, lifts: dict) -> float:
    comp, base = e["comp"], e["base"]
    v, m = e["vals"][c], e["mins"]
    b = _per36(v[base], m[base])
    age = np.array([e["i"] - j for j in comp], dtype=float)
    w = 0.5 ** ((age - 1) / half) if half > 0 else np.ones(len(comp))
    vv = v[comp].copy()
    if mode == "minutes":
        lw = [max(LW_FLOOR, float(np.prod([EXTRA_OUT_BY_POS.get(frozenset(e["grp"] + g), EXTRA_OUT_W) for g in eg])))
              for eg in e["extra_grp"]]
        w = w * np.array(lw)
    elif mode == "adjust":
        vv = vv / np.array([np.prod([lifts.get((c, e["grp"], g), 1.0) for g in eg]) for eg in e["extra_grp"]])
    s = float(np.sum(vv * w) / np.sum(m[comp] * w) * 36)
    return (s * w.sum() + b * k) / (w.sum() + k)


def score(evs: list[dict], lifts: dict, half: float, k: float, mode: str, nmin: int = 1, nmax: int = 999) -> dict:
    res = {}
    for c in COLS:
        se, n = 0.0, 0
        for e in evs:
            if not nmin <= e["n"] <= nmax:
                continue
            p = predict(e, c, half, k, mode, lifts) * e["mins"][e["i"]] / 36
            se += (e["vals"][c][e["i"]] - p) ** 2
            n += 1
        res[c] = se / max(n, 1)
        res["n"] = n
    return res


def main() -> None:
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    trk = load_tracking()
    grp = dict(position_groups(positions()).iter_rows())
    evs = {s: events(s, pg, trk, grp) for s in SEASONS}
    lifts = {s: lift_table(evs[o]) for s, o in zip(SEASONS, SEASONS[::-1])}  # fit on the other season
    for s in SEASONS:
        print(f"\n== {s}: {len(evs[s])} new-role games")
        for (c, me, them), v in sorted(lifts[s].items()):
            if c == "reb_chances":
                print(f"   lift {c} {me}<-{them} out: {v:.3f}")
    rows = []
    buckets = ((1, 2), (3, 6), (7, 999))
    grid = [(mode, half, k) for mode in ("none", "minutes") for half in (1.5, 3.0, 6.0, 10.0, 0) for k in (1.0, 3.0, 6.0)]
    for lo, hi in buckets:
        for mode, half, k in grid:
            tot, n = {c: 0.0 for c in COLS}, 0
            for s in SEASONS:
                r = score(evs[s], lifts[s], half, k, mode, lo, hi)
                for c in COLS:
                    tot[c] += r[c] * r["n"]
                n += r["n"]
            rows.append({"games": f"{lo}-{hi}", "mode": mode, "half": half, "k": k, "n": n, **{c: tot[c] / n for c in COLS}})
    df = pl.DataFrame(rows)
    with pl.Config(tbl_rows=100, tbl_cols=20):
        for lo, hi in buckets:
            d = df.filter(pl.col("games") == f"{lo}-{hi}")
            ref = d.filter(pl.col("mode") == "minutes", pl.col("half") == 3.0, pl.col("k") == 3.0).row(0, named=True)
            print(f"\nmatching games {lo}-{hi} (n={ref['n']}); MSE relative to current (minutes, half 3, k 3)")
            print(d.with_columns(*[(pl.col(c) / ref[c]).round(4).alias(c) for c in COLS],
                                 ((pl.col("reb_chances") / ref["reb_chances"] + pl.col("potential_ast") / ref["potential_ast"]
                                   + pl.col("pts") / ref["pts"]) / 3).round(4).alias("avg")).sort("avg").head(10))

if __name__ == "__main__":
    main()
