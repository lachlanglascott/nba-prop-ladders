"""Opponent by position / similar player: expected vs actual, is it predictive, is it priced?

For each player-game (10+ min): expected stat = his own per-minute rate over his previous 40 games
x actual minutes (so this isolates rate, not minutes). Opponent factor, as at the game date (only
earlier games), at three levels of "similar player":

    team   - everyone vs this opponent
    grp    - G / W / B (what spots/matchup use today)
    arch   - archetype: lead guard / scoring guard / wing / forward / stretch big / rim big

Each level shrinks toward the level above (arch -> grp -> team -> 1.0) with pseudo-games K.
Last season's end-of-season factor (regressed) is the prior for early-season games.

Tests:
  1. predictive: slope of actual/expected on the as-at factor (1.0 = the factor is fully real)
  2. priced:    on the market backtest (Sportsbet X+ rungs), does log(factor) explain hits beyond
                logit(book)? and ROI of overs where the factor is strong.

    uv run python -m nba_ladders.opp_position [--player NAME]
"""

import argparse
from functools import lru_cache

import numpy as np
import polars as pl

from nba_ladders import config

STATS = ("pts", "reb", "ast")
WIN = 40  # games in the player's own trailing rate
MIN_PRIOR = 10  # games before his rate counts
LEVELS = ("team", "grp", "arch")
K = {"team": 15.0, "grp": 25.0, "arch": 25.0}  # pseudo-games toward the level above
PRIOR_REGRESS = 0.5  # last season's factor regressed this far to the level above
K_PRIOR = 15.0  # pseudo-games of last season in this season's factor


def archetype(grp: str, ast36: float, reb36: float, fg3a36: float) -> str:
    if grp == "G":
        return "lead_guard" if ast36 >= 6.5 else "scoring_guard"
    if grp == "B":
        return "stretch_big" if fg3a36 >= 3.0 else "rim_big"
    return "forward" if reb36 >= 8.0 else "wing"


def player_games() -> pl.DataFrame:
    """Player-games with as-at expected stats and archetype."""
    from nba_ladders.absences import position_groups
    from nba_ladders.scenarios import positions

    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(pl.col("minutes") > 0)
    pg = pg.join(position_groups(positions()), on="player_id", how="left").with_columns(pl.col("grp").fill_null("W"))
    pg = pg.sort("player_id", "game_date")
    roll = lambda c: pl.col(c).rolling_sum(WIN, min_samples=1).shift(1).over("player_id")
    pg = pg.with_columns(
        *[roll(c).alias(f"_{c}") for c in (*STATS, "minutes", "fg3a")],
        pl.int_range(pl.len()).over("player_id").alias("n_prior"),
    )
    pg = pg.with_columns(*[(pl.col(f"_{s}") / pl.col("_minutes") * pl.col("minutes")).alias(f"e_{s}") for s in STATS],
                         *[(pl.col(f"_{c}") / pl.col("_minutes") * 36).alias(f"{c}36") for c in ("ast", "reb", "fg3a")])
    pg = pg.filter(pl.col("minutes") >= 10, pl.col("n_prior") >= MIN_PRIOR)
    arch = [archetype(g, a, r, f) for g, a, r, f in pg.select("grp", "ast36", "reb36", "fg3a36").iter_rows()]
    return pg.with_columns(pl.Series("arch", arch)).with_columns(pl.lit("all").alias("team_lvl"))


def _cum(pg: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per (season, opponent, keys, date): actual and expected sums over strictly earlier dates."""
    day = pg.group_by("season_id", "opponent", *keys, "game_date").agg(
        pl.col("game_id").n_unique().alias("g"), *[pl.col(s).sum().alias(f"a_{s}") for s in STATS],
        *[pl.col(f"e_{s}").sum() for s in STATS]).sort("game_date")
    cols = ["g", *[f"a_{s}" for s in STATS], *[f"e_{s}" for s in STATS]]
    over = ["season_id", "opponent", *keys]
    return day.with_columns(*[(pl.col(c).cum_sum().over(over) - pl.col(c)).alias(f"c{c}") for c in cols])


def _final(pg: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """End-of-season raw ratio per (season, opponent, keys)."""
    return pg.group_by("season_id", "opponent", *keys).agg(
        pl.col("game_id").n_unique().alias("g"),
        *[(pl.col(s).sum() / pl.col(f"e_{s}").sum()).alias(f"r_{s}") for s in STATS])


def as_at_factors(pg: pl.DataFrame) -> pl.DataFrame:
    """pg plus f_<level>_<stat>: shrunk as-at opponent factor at each level."""
    keys = {"team": ["team_lvl"], "grp": ["grp"], "arch": ["grp", "arch"]}
    out = pg
    parent = {s: pl.lit(1.0) for s in STATS}
    for lvl in LEVELS:
        c = _cum(pg, keys[lvl])
        fin = _final(pg, keys[lvl]).with_columns(
            (pl.col("season_id").cast(pl.Int64) + 1).cast(pl.Utf8).alias("season_id"))
        fin = fin.rename({f"r_{s}": f"prev_{s}" for s in STATS} | {"g": "prev_g"})
        out = out.join(c.select("season_id", "opponent", *keys[lvl], "game_date", "cg",
                                *[f"ca_{s}" for s in STATS], *[f"ce_{s}" for s in STATS]),
                       on=["season_id", "opponent", *keys[lvl], "game_date"], how="left")
        out = out.join(fin, on=["season_id", "opponent", *keys[lvl]], how="left")
        new = []
        for s in STATS:
            par = parent[s]
            # last season, raw shrunk to parent by its games, then regressed
            prev = (pl.col(f"prev_{s}") * pl.col("prev_g") + par * K[lvl]) / (pl.col("prev_g") + K[lvl])
            prior = (par + (1 - PRIOR_REGRESS) * (prev.fill_nan(None) - par)).fill_null(par)
            raw = (pl.col(f"ca_{s}") / pl.col(f"ce_{s}")).fill_nan(None)
            # this season's games: shrink toward (prior blended with parent)
            n = pl.col("cg").fill_null(0)
            f = ((raw.fill_null(1.0) * n + prior * K_PRIOR + par * K[lvl]) / (n + K_PRIOR + K[lvl]))
            new.append(f.alias(f"f_{lvl}_{s}"))
        out = out.with_columns(new).drop("cg", "prev_g", *[f"{p}_{s}" for p in ("ca", "ce", "prev") for s in STATS])
        parent = {s: pl.col(f"f_{lvl}_{s}") for s in STATS}
    return out


def table(as_of, season_id: str = "22026", pg: pl.DataFrame | None = None) -> pl.DataFrame:
    """As-at factors for every opponent x group (x archetype) on `as_of`, from earlier games only.

    Adds one empty placeholder row per (opponent, archetype) dated as_of, so the as-at machinery
    gives exactly what a game that day would have seen."""
    pg = player_games() if pg is None else pg
    pg = pg.filter(pl.col("game_date") < as_of)
    arch_grp = {"lead_guard": "G", "scoring_guard": "G", "wing": "W", "forward": "W", "stretch_big": "B", "rim_big": "B"}
    opps = pg.filter(pl.col("season_id") == season_id)["opponent"].unique().to_list()
    ph = pl.DataFrame([{"season_id": season_id, "opponent": o, "grp": g, "arch": a, "game_date": as_of,
                        "game_id": "placeholder", "team_lvl": "all", **{s: 0 for s in STATS}, **{f"e_{s}": 0.0 for s in STATS}}
                       for o in opps for a, g in arch_grp.items()])
    full = pl.concat([pg, ph], how="diagonal_relaxed")
    return as_at_factors(full).filter(pl.col("game_id") == "placeholder").select(
        "opponent", "grp", "arch", *[f"f_{lvl}_{s}" for lvl in LEVELS for s in STATS])


def predictive(df: pl.DataFrame, season: str | None = None) -> pl.DataFrame:
    """Per level and stat: weighted slope of (actual/expected - 1) on (factor - 1), and corr."""
    d = df if season is None else df.filter(pl.col("season_id") == season)
    rows = []
    for s in STATS:
        dd = d.filter(pl.col(f"e_{s}") > 0.5)
        y = (dd[s] / dd[f"e_{s}"] - 1).to_numpy()
        w = dd[f"e_{s}"].to_numpy()
        for lvl in LEVELS:
            x = (dd[f"f_{lvl}_{s}"] - 1).to_numpy()
            xm, ym = np.average(x, weights=w), np.average(y, weights=w)
            slope = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
            # incremental over the level above (arch vs grp etc)
            rows.append({"stat": s, "level": lvl, "slope": slope, "sd_factor": float(np.sqrt(np.average((x - xm) ** 2, weights=w))),
                         "corr": float(np.corrcoef(x, y)[0, 1]), "n": len(y)})
    return pl.DataFrame(rows)


def incremental(df: pl.DataFrame) -> pl.DataFrame:
    """Does the finer level add over the coarser? slope of residual-after-coarse on (fine/coarse - 1)."""
    rows = []
    pairs = (("team", None), ("grp", "team"), ("arch", "grp"))
    for s in STATS:
        dd = df.filter(pl.col(f"e_{s}") > 0.5)
        w = dd[f"e_{s}"].to_numpy()
        for fine, coarse in pairs:
            base = dd[f"f_{coarse}_{s}"].to_numpy() if coarse else np.ones(len(dd))
            y = (dd[s] / (dd[f"e_{s}"] * base) - 1).to_numpy()
            x = (dd[f"f_{fine}_{s}"].to_numpy() / base) - 1
            xm, ym = np.average(x, weights=w), np.average(y, weights=w)
            b = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
            res = y - ym - b * (x - xm)
            se = np.sqrt(np.sum(w**2 * (x - xm) ** 2 * res**2)) / np.sum(w * (x - xm) ** 2)
            rows.append({"stat": s, "fine": fine, "over": coarse or "-", "slope": b, "se": se})
    return pl.DataFrame(rows)


# ---- market ---------------------------------------------------------------------------------------
def market_join(df: pl.DataFrame) -> pl.DataFrame:
    mb = pl.read_parquet(config.DATA_PROCESSED / "market_backtest.parquet")
    f = df.select("player_id", "game_date", "grp", "arch", "minutes", *[f"e_{s}" for s in STATS],
                  *[f"f_{l}_{s}" for l in LEVELS for s in STATS])
    m = mb.join(f, on=["player_id", "game_date"], how="inner")
    # P+R: expected-weighted factor
    for lvl in LEVELS:
        pr = (pl.col(f"f_{lvl}_pts") * pl.col("e_pts") + pl.col(f"f_{lvl}_reb") * pl.col("e_reb")) / (pl.col("e_pts") + pl.col("e_reb"))
        m = m.with_columns(pl.when(pl.col("stat") == "pts").then(pl.col(f"f_{lvl}_pts"))
                           .when(pl.col("stat") == "reb").then(pl.col(f"f_{lvl}_reb"))
                           .when(pl.col("stat") == "ast").then(pl.col(f"f_{lvl}_ast"))
                           .otherwise(pr).alias(f"f_{lvl}"))
    return m


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def market_coef(m: pl.DataFrame, lvl: str) -> dict:
    """hit ~ a + b logit(p_book) + c log(factor); c>0 = book under-reacts to the opponent spot."""
    X = np.column_stack([np.ones(m.height), _logit(m["p_book"].to_numpy()), np.log(m[f"f_{lvl}"].to_numpy())])
    beta, se = logit_fit(X, m["hit"].to_numpy(), pd_codes(m["event_id"]))
    return {"level": lvl, "c": round(beta[2], 3), "se": round(se[2], 3), "b_book": round(beta[1], 3)}


def logit_fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton, standard errors clustered by game."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    sc = X * (y - p)[:, None]
    G = np.zeros((groups.max() + 1, X.shape[1]))
    np.add.at(G, groups, sc)
    Hi = np.linalg.inv(H)
    return b, np.sqrt(np.diag(Hi @ (G.T @ G) @ Hi))


def pd_codes(s: pl.Series) -> np.ndarray:
    return s.cast(pl.Categorical).to_physical().to_numpy()


def roi_by_bucket(m: pl.DataFrame, lvl: str, stat: str | None = None, cuts=(0.94, 0.97, 1.03, 1.06)) -> pl.DataFrame:
    d = m if stat is None else m.filter(pl.col("stat") == stat)
    b = d.with_columns(pl.col(f"f_{lvl}").cut(list(cuts)).alias("bucket"))
    return b.group_by("bucket").agg(
        pl.len().alias("rungs"), pl.col("event_id").n_unique().alias("games"),
        (pl.col("hit") - pl.col("p_book")).mean().alias("hit_minus_book"),
        (pl.col("hit") * pl.col("price") - 1).mean().alias("roi")).sort("bucket")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player")
    args = ap.parse_args()
    df = as_at_factors(player_games())
    df.write_parquet(config.DATA_PROCESSED / "opp_position.parquet")
    with pl.Config(tbl_rows=60, tbl_hide_column_data_types=True, float_precision=3, tbl_width_chars=200):
        if args.player:
            print(df.filter(pl.col("player_name").str.contains(args.player)).sort("game_date").tail(12).select(
                "game_date", "opponent", "arch", "minutes", *[c for s in STATS for c in (s, f"e_{s}")],
                "f_arch_pts", "f_arch_reb", "f_arch_ast"))
            return
        print("Predictive (all seasons): slope of actual/expected on the as-at factor")
        print(predictive(df.filter(pl.col("season_id") >= "22021")))
        print("\nIncremental: does the finer level add beyond the coarser one?")
        print(incremental(df.filter(pl.col("season_id") >= "22021")))
        m = market_join(df)
        print(f"\nMarket rungs joined: {m.height}")
        for lvl in LEVELS:
            print(market_coef(m, lvl))
        for st in (None, "pts", "reb", "ast", "pts+reb"):
            print(f"\nROI by arch factor, stat={st or 'all'}")
            print(roi_by_bucket(m, "arch", st))


if __name__ == "__main__":
    main()


# ---- angle H: gate / boost for spots (docs/opp_position.md §3-4) -----------------------------------
# Sportsbet 2024-26, ROI by the opponent's G/W/B factor for the bet's stat (player-games, both seasons agree):
#   bigs (starting or backup) pts / reb / P+R: factor < 1.02 -> -12 to -14%; >= 1.02 -> P+R -0.6%, pts -3%, reb -7%
#   guards / wings assists: < 0.97 -> -13% / -9%; >= 1.02 -> -5.5% / 0%; guards' pts and G/W reb: flat (no action)
#   wings' pts: < 0.97 -> -12.3% vs -7 to -9% above
#   starter-out backup bigs: < 1.02 -> -15%; >= 1.02 -> +8%;  assist luck x opp ast >= 1.04: +22% (83 games)
BIG_CUT = 1.02  # bigs' P/R: below = pass, at or above = boost
AST_LOW, AST_HIGH = 0.97, 1.02  # guards' / wings' assists: below = graded down, at or above = boost
LUCK_AST = 1.04  # assist bounce x opponent ast factor
WING_PTS_LOW = 0.97  # wings' points below this: graded down (-12.3% vs -7 to -9%, both seasons; guards' pts flat)
BOOST_C = 0.05  # confidence added to a boosted bet
LEG = {"p": "pts", "r": "reb", "a": "ast"}


def bet_stats(b: dict) -> list[str]:
    """Stats a spots bet depends on: '15p/10r' -> [pts, reb]; '17+ pts' -> [pts]; '20+ pts+reb' -> [pts, reb]."""
    if b["kind"] == "SGM":
        return [LEG[x[-1]] for x in b["bet"].split("/")]
    return b["bet"].split("+ ", 1)[1].split("+")


def bet_factor(fac: dict, stats: list[str]) -> float:
    """Opponent factor for a bet: geometric mean over its stats (1.0 when unknown)."""
    v = [fac.get(s, 1.0) for s in stats]
    return float(np.prod(v) ** (1 / len(v))) if v else 1.0


def angle_h(fac: dict, b: dict, kinds: set, grp: str, bounce_stats: set) -> tuple[str, str]:
    """(action, note) for one bet. action: 'gate' (pass), 'down' (one grade lower), 'boost', or ''.

    Bigs: the pts/reb legs decide, starting or backup, any spot. Guards / wings: only assist legs count."""
    st = bet_stats(b)
    if grp == "B" and (pr := [s for s in st if s in ("pts", "reb")]):
        f, lab = bet_factor(fac, pr), "/".join(pr)
        if f < BIG_CUT:
            return "gate", f"opp spot: bad for bigs' {lab} ({f:.2f} < {BIG_CUT}), pass (angle H)"
        return "boost", f"opp spot: good for bigs' {lab} ({f:.2f}), +confidence (angle H)"
    if grp in ("G", "W") and "ast" in st:
        f = fac.get("ast", 1.0)
        if "ast" in bounce_stats and f >= LUCK_AST:
            return "boost", f"assist luck x good opp ast spot ({f:.2f}) (angle H)"
        if f < AST_LOW:
            return "down", f"opp spot: bad for {grp} assists ({f:.2f}), graded down (angle H)"
        if f >= AST_HIGH and kinds & {"role", "bounce"}:
            return "boost", f"opp spot: good for {grp} assists ({f:.2f}), +confidence (angle H)"
    if grp == "W" and "pts" in st and (f := fac.get("pts", 1.0)) < WING_PTS_LOW:
        return "down", f"opp spot: bad for wings' pts ({f:.2f}), graded down (angle H)"
    return "", ""


def line(fac: dict, opp: str, grp: str, pace: float) -> str:
    """Plain words, same factors angle H uses: 'DET vs wings: pts -4%, reb +0%, ast -9%; pace -0%'."""
    name = {"G": "guards", "W": "wings", "B": "bigs"}[grp]
    bits = ", ".join(f"{s} {(fac.get(s, 1.0) - 1) * 100:+.0f}%" for s in STATS)
    return f"{opp} vs {name}: {bits}; pace {(pace - 1) * 100:+.0f}%"


# Weight on each factor in projections: rate x factor ** w. Slope of actual/expected on the as-at factor,
# 2020-21 to 2026-27, +-0.05-0.08 (docs/opp_position.md §1, §4).
OPP_W = {"G": {"pts": 0.93, "reb": 0.82, "ast": 0.98},
         "W": {"pts": 0.92, "reb": 0.82, "ast": 0.82},
         "B": {"pts": 1.12, "reb": 0.95, "ast": 0.55}}


@lru_cache(maxsize=8)
def tonight(as_of) -> dict:
    """{(opponent, grp): {pts, reb, ast}} as at as_of: the G/W/B factors the angle was tested on."""
    t = table(as_of).unique(["opponent", "grp"])
    return {(r["opponent"], r["grp"]): {s: r[f"f_grp_{s}"] for s in STATS} for r in t.iter_rows(named=True)}
