"""Usage redistribution when teammates sit: league prior + player-specific absorption.

When a rotation teammate T misses a game, the volume he usually carries (usage = FGA + 0.44 FTA + TOV,
assists, rebounds per game) goes to the players who are left. How much each player picks up is
player-specific (Edgecombe took +4 shots a game without Embiid in 2025-26), so:

  multiplier on player i's per-minute stat s, tonight = 1 + sum over absent T of effect(i, T, s)
  effect(i, T, s) = (sum of i's residuals in past games T missed + K * prior(i, T, s)) / (n + K)
  prior(i, T, s)  = beta_s[same/diff position] * T's volume / 100 * (i's own rate / league rate)

Residuals are vs i's absence-neutral trailing rate (his history games divided by their own prior
multiplier), using only earlier games. Pairs count across seasons and lineups: what he did without
T last season counts even if the roster has changed since. Absence = played 3+ of the team's last 10
games at 20+ min and missed this one (as-at, so traded / long-injured players drop out).

    uv run python -m nba_ladders.usage            # fit prior on 2019-24, backtest 2024-27
    uv run python -m nba_ladders.usage --player "VJ Edgecombe" --out "Joel Embiid,LeBron James"
"""

import argparse
from datetime import date
from functools import lru_cache

import numpy as np
import polars as pl

from nba_ladders import config

STATS = ("pts", "ast", "reb")
VOL = {"pts": "u", "ast": "ast", "reb": "reb"}  # the teammate volume that frees each stat
ROT_WINDOW, ROT_MIN_GP, ROT_MPG = 10, 3, 20.0
VOL_GAMES = 20  # T's volume: his last 20 games played
WIN = 40  # i's trailing rate
MIN_PRIOR = 10
FIT_BEFORE = "22024"  # prior fitted on seasons before this; backtest on this and later
# Pseudo-games of prior in a pair effect. Tuned on 2024-25, deviance vs plain trailing rate (league prior alone:
# pts -408, ast -61, reb -2): K=20 pts -126 / 40 -423 / 80 -490 / 160 -478 / 320 -452; ast best at 80 (-119);
# reb ~0 at any K. Held out 2025-26 at these K: pts -537 (prior -433), ast -118 (-68). Single games are noisy
# (pts +-45%) next to true pair effects (+-5%), hence the heavy shrinkage.
K_PAIR = {"pts": 80.0, "ast": 80.0, "reb": 160.0}
CLIP = (0.75, 1.5)
REB_FLOOR = 5.0  # rebounds per game a typical absentee "uses"; only the excess is freed for teammates (2023-27 tracking check)


def _frame() -> pl.DataFrame:
    from nba_ladders.absences import player_groups

    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    pg = pg.join(player_groups(), on="player_id", how="left").with_columns(pl.col("grp").fill_null("W"))
    return pg.with_columns((pl.col("fga") + 0.44 * pl.col("fta") + pl.col("tov")).alias("u"))


def absences(pg: pl.DataFrame) -> pl.DataFrame:
    """(team, game_id, game_date, mate_id, mate_grp, vol_u, vol_ast, vol_reb): rotation teammates missing, as-at."""
    # window within the season: last season's departed players aren't "absent" in October
    tg = pg.select("season_id", "team", "game_id", "game_date").unique().sort("team", "game_date") \
        .with_columns(pl.int_range(pl.len()).over("season_id", "team").alias("k"))
    app = pg.filter(pl.col("minutes") > 0).join(tg, on=["season_id", "team", "game_id", "game_date"]) \
        .select("season_id", "team", "k", "player_id", "minutes")
    win = pl.concat([app.with_columns(pl.col("k") + o) for o in range(1, ROT_WINDOW + 1)])
    rot = win.group_by("season_id", "team", "k", "player_id").agg(pl.len().alias("n"), pl.col("minutes").mean().alias("m")) \
        .filter(pl.col("n") >= ROT_MIN_GP, pl.col("m") >= ROT_MPG)
    miss = rot.join(app.select("season_id", "team", "k", "player_id"), on=["season_id", "team", "k", "player_id"], how="anti") \
        .join(tg, on=["season_id", "team", "k"]).rename({"player_id": "mate_id"})
    # T's volume per game, his last VOL_GAMES played games before the date (any team)
    pl_ = pg.filter(pl.col("minutes") > 0).sort("player_id", "game_date")
    vol = pl_.with_columns(*[pl.col(c).rolling_mean(VOL_GAMES, min_samples=5).over("player_id").alias(f"vol_{c}")
                             for c in ("u", "ast", "reb")]).select("player_id", "game_date", "grp", "vol_u", "vol_ast", "vol_reb")
    vol = vol.rename({"player_id": "mate_id", "grp": "mate_grp", "game_date": "vd"}).sort("vd")
    miss = miss.sort("game_date").join_asof(vol, left_on="game_date", right_on="vd", by="mate_id", strategy="backward",
                                            allow_exact_matches=False)
    return miss.drop_nulls("vol_u").select("team", "game_id", "game_date", "mate_id", "mate_grp", "vol_u", "vol_ast", "vol_reb")


def league_rates(pg: pl.DataFrame) -> dict:
    d = pg.filter(pl.col("minutes") >= 15)
    return {s: float(d[s].sum() / d["minutes"].sum()) for s in STATS}


def player_rows(pg: pl.DataFrame) -> pl.DataFrame:
    """Player-games (10+ min, 10+ prior games) with trailing per-minute rates (plain)."""
    pg = pg.filter(pl.col("minutes") > 0).sort("player_id", "game_date")
    roll = lambda c: pl.col(c).rolling_sum(WIN, min_samples=1).shift(1).over("player_id")
    pg = pg.with_columns(*[roll(c).alias(f"_{c}") for c in (*STATS, "minutes")],
                         pl.int_range(pl.len()).over("player_id").alias("n_prior"))
    return pg.with_columns(*[(pl.col(f"_{s}") / pl.col("_minutes")).alias(f"r_{s}") for s in STATS])


def prior_x(rows: pl.DataFrame, ab: pl.DataFrame, lg: dict) -> pl.DataFrame:
    """Per (player-game, absent mate): prior regressors x_same_s, x_diff_s = mate volume/100 x own-rate tilt."""
    j = rows.select("player_id", "team", "game_id", "grp", *[f"r_{s}" for s in STATS]).join(ab, on=["team", "game_id"])
    j = j.filter(pl.col("mate_id") != pl.col("player_id"))
    cols = []
    for s in STATS:
        tilt = (pl.col(f"r_{s}") / lg[s]).fill_null(1.0).clip(0.2, 3.0)
        x = pl.col(f"vol_{VOL[s]}") / 100 * tilt
        if s == "reb":  # rebound share: only a big rebounder's absence frees boards (rates +2-6% at 8+ rpg, ~0 below), no tilt
            x = (pl.col("vol_reb") - REB_FLOOR).clip(0, None) / 100
        same = pl.col("grp") == pl.col("mate_grp")
        cols += [pl.when(same).then(x).otherwise(0.0).alias(f"xs_{s}"), pl.when(same).then(0.0).otherwise(x).alias(f"xd_{s}")]
    return j.with_columns(cols)


def fit_prior(rows: pl.DataFrame, px: pl.DataFrame) -> dict:
    """beta_s (same, diff position): weighted least squares of actual/expected - 1 on summed x, seasons < FIT_BEFORE."""
    agg = px.group_by("player_id", "game_id").agg(*[pl.col(f"{p}_{s}").sum() for p in ("xs", "xd") for s in STATS])
    d = rows.filter(pl.col("season_id") < FIT_BEFORE, pl.col("minutes") >= 10, pl.col("n_prior") >= MIN_PRIOR) \
        .join(agg, on=["player_id", "game_id"], how="left").fill_null(0.0)
    beta = {}
    for s in STATS:
        e = (d[f"r_{s}"] * d["minutes"]).to_numpy()
        ok = e > 0.5
        y = d[s].to_numpy()[ok] / e[ok] - 1
        X = np.column_stack([np.ones(ok.sum()), d[f"xs_{s}"].to_numpy()[ok], d[f"xd_{s}"].to_numpy()[ok]])
        w = e[ok]
        b = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
        beta[s] = {"same": float(b[1]), "diff": float(b[2])}
    return beta


def with_multipliers(rows: pl.DataFrame, px: pl.DataFrame, beta: dict, k_pair: dict | float = K_PAIR) -> pl.DataFrame:
    """rows + m_prior_s (prior multiplier), e_neutral_s (absence-neutral trailing rate x minutes) and m_pair_s
    (player-specific multiplier from strictly earlier games)."""
    kp = k_pair if isinstance(k_pair, dict) else {s: k_pair for s in STATS}
    px = px.with_columns(*[(beta[s]["same"] * pl.col(f"xs_{s}") + beta[s]["diff"] * pl.col(f"xd_{s}")).alias(f"p_{s}") for s in STATS])
    mp = px.group_by("player_id", "game_id").agg(*[pl.col(f"p_{s}").sum().alias(f"mp_{s}") for s in STATS], pl.len().alias("n_abs"))
    r = rows.join(mp, on=["player_id", "game_id"], how="left").with_columns(
        *[(1 + pl.col(f"mp_{s}").fill_null(0.0)).clip(*CLIP).alias(f"m_prior_{s}") for s in STATS], pl.col("n_abs").fill_null(0))
    # absence-neutral trailing rate: each history game divided by its own prior multiplier
    r = r.sort("player_id", "game_date").with_columns(
        *[(pl.col(s) / pl.col(f"m_prior_{s}")).rolling_sum(WIN, min_samples=1).shift(1).over("player_id").alias(f"_n_{s}") for s in STATS])
    r = r.with_columns(*[(pl.col(f"_n_{s}") / pl.col("_minutes") * pl.col("minutes")).alias(f"e_neutral_{s}") for s in STATS])
    # pair residuals: in games mate T missed, i's actual / (neutral expected) - 1 - prior of the OTHER absentees
    g = r.select("player_id", "game_id", "game_date", "minutes", *STATS, *[f"e_neutral_{s}" for s in STATS],
                 *[f"m_prior_{s}" for s in STATS])
    pj = px.select("player_id", "game_id", "mate_id", *[f"p_{s}" for s in STATS]).join(g, on=["player_id", "game_id"])
    pj = pj.with_columns(*[pl.when(pl.col(f"e_neutral_{s}") > 0.05).then(pl.col(s) / pl.col(f"e_neutral_{s}") - 1 - (pl.col(f"m_prior_{s}") - 1 - pl.col(f"p_{s}"))).otherwise(0.0).alias(f"res_{s}")
                           for s in STATS]).filter(pl.col("e_neutral_pts").is_not_null())
    # as-at per pair: sums over strictly earlier dates (weight = expected volume so tiny-minute games count less)
    pj = pj.sort("game_date").with_columns(*[pl.when(pl.col(f"e_neutral_{s}") > 0.05).then(pl.col(f"e_neutral_{s}")).otherwise(0.0).fill_nan(0).alias(f"w_{s}") for s in STATS])
    over = ["player_id", "mate_id"]
    cum = []
    for s in STATS:
        wr = pl.col(f"w_{s}") * pl.col(f"res_{s}").fill_nan(0)
        cum += [(wr.cum_sum().over(over) - wr).alias(f"S_{s}"), (pl.col(f"w_{s}").cum_sum().over(over) - pl.col(f"w_{s}")).alias(f"W_{s}")]
    pj = pj.with_columns(cum)
    # effect = (sum w res + K*wbar*prior) / (sum w + K*wbar): K pseudo-games at this game's weight
    eff = pj.with_columns(*[((pl.col(f"S_{s}") + kp[s] * pl.col(f"w_{s}") * pl.col(f"p_{s}"))
                             / (pl.col(f"W_{s}") + kp[s] * pl.col(f"w_{s}"))).fill_nan(None).fill_null(pl.col(f"p_{s}")).alias(f"eff_{s}")
                           for s in STATS])
    me = eff.group_by("player_id", "game_id").agg(*[pl.col(f"eff_{s}").sum().alias(f"me_{s}") for s in STATS])
    r = r.join(me, on=["player_id", "game_id"], how="left")
    return r.with_columns(*[(1 + pl.col(f"me_{s}").fill_null(0.0)).clip(*CLIP).alias(f"m_pair_{s}") for s in STATS])


def deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-6, None)
    return float(2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu)))


def evaluate(r: pl.DataFrame, season: str) -> dict:
    """Poisson deviance of three projections (actual minutes): plain trailing rate; neutral x prior; neutral x pair."""
    d = r.filter(pl.col("season_id") == season, pl.col("minutes") >= 10, pl.col("n_prior") >= MIN_PRIOR,
                 pl.col("e_neutral_pts").is_not_null())
    out = {}
    for s in STATS:
        y = d[s].to_numpy().astype(float)
        plain = (d[f"r_{s}"] * d["minutes"]).to_numpy()
        pri = (d[f"e_neutral_{s}"] * d[f"m_prior_{s}"]).to_numpy()
        pair = (d[f"e_neutral_{s}"] * d[f"m_pair_{s}"]).to_numpy()
        base = deviance(y, plain)
        a = d["n_abs"].to_numpy() > 0
        out[s] = {"plain": round(base), "prior": round(deviance(y, pri) - base), "pair": round(deviance(y, pair) - base),
                  "with_absence_games": int(a.sum()),
                  "prior_abs": round(deviance(y[a], pri[a]) - deviance(y[a], plain[a])),
                  "pair_abs": round(deviance(y[a], pair[a]) - deviance(y[a], plain[a]))}
    return out


@lru_cache(maxsize=2)
def _state():
    pg = _frame()
    rows, ab, lg = player_rows(pg), absences(pg), league_rates(pg)
    px = prior_x(rows, ab, lg)
    return pg, rows, ab, lg, px, fit_prior(rows, px)


def tonight(pid: int, out_ids: set, as_of: date, k_pair: dict = K_PAIR) -> dict:
    """Per-minute multipliers {pts, ast, reb} for player pid tonight with `out_ids` sitting, vs his normal lineup,
    plus per-mate detail. Uses only games before as_of."""
    pg, rows, _, lg, _, beta = _state()
    hist = rows.filter(pl.col("player_id") == pid, pl.col("game_date") < as_of)
    if hist.is_empty() or not out_ids:
        return {"mult": {s: 1.0 for s in STATS}, "mates": []}
    me = hist.tail(1).row(0, named=True)
    rate = {s: (hist[s].tail(WIN).sum() / max(hist["minutes"].tail(WIN).sum(), 1)) for s in STATS}
    grp = me["grp"]
    pr = pg.filter(pl.col("minutes") > 0, pl.col("game_date") < as_of).sort("game_date")
    _, rows_m = with_multipliers_cached()
    pairs = _pair_history(rows_m, pid, as_of)
    mates, tot = [], {s: 0.0 for s in STATS}
    for t in out_ids:
        th = pr.filter(pl.col("player_id") == t).tail(VOL_GAMES)
        if th.height < 5:
            continue
        vol = {"u": th["u"].mean(), "ast": th["ast"].mean(), "reb": th["reb"].mean()}
        same = th["grp"][-1] == grp
        det = {"mate": th["player_name"][-1], "games_without": 0}
        for s in STATS:
            tilt = min(max(rate[s] / lg[s], 0.2), 3.0)
            x = max(vol["reb"] - REB_FLOOR, 0) / 100 if s == "reb" else vol[VOL[s]] / 100 * tilt
            p = beta[s]["same" if same else "diff"] * x
            ph = pairs.filter(pl.col("mate_id") == t)
            W = float(ph[f"w_{s}"].sum()) if ph.height else 0.0
            S = float((ph[f"w_{s}"] * ph[f"res_{s}"].fill_nan(0)).sum()) if ph.height else 0.0
            wbar = rate[s] * float(hist["minutes"].tail(10).mean())
            e = (S + k_pair[s] * wbar * p) / (W + k_pair[s] * wbar) if (W + k_pair[s] * wbar) > 0 else p
            e = e if np.isfinite(e) else p
            tot[s] += e
            det[s] = {"prior": p, "own": (S / W) if W else None, "effect": e}
            det["games_without"] = ph.height
        mates.append(det)
    return {"mult": {s: float(np.clip(1 + tot[s], *CLIP)) for s in STATS}, "mates": mates}


@lru_cache(maxsize=1)
def with_multipliers_cached():
    _, rows, _, _, px, beta = _state()
    return beta, with_multipliers(rows, px, beta, K_PAIR)


def _pair_history(r: pl.DataFrame, pid: int, as_of: date) -> pl.DataFrame:
    """Player pid's past games with each absent mate: residual vs neutral and weight (before as_of)."""
    *_, px, beta = _state()
    px = px.filter(pl.col("player_id") == pid)
    px = px.with_columns(*[(beta[s]["same"] * pl.col(f"xs_{s}") + beta[s]["diff"] * pl.col(f"xd_{s}")).alias(f"p_{s}") for s in STATS])
    g = r.filter(pl.col("player_id") == pid, pl.col("game_date") < as_of).select(
        "game_id", "game_date", *STATS, *[f"e_neutral_{s}" for s in STATS], *[f"m_prior_{s}" for s in STATS])
    pj = px.select("game_id", "mate_id", *[f"p_{s}" for s in STATS]).join(g, on="game_id")
    return pj.with_columns(*[pl.when(pl.col(f"e_neutral_{s}") > 0.05).then(pl.col(s) / pl.col(f"e_neutral_{s}") - 1 - (pl.col(f"m_prior_{s}") - 1 - pl.col(f"p_{s}"))).otherwise(0.0).alias(f"res_{s}")
                             for s in STATS], *[pl.when(pl.col(f"e_neutral_{s}") > 0.05).then(pl.col(f"e_neutral_{s}")).otherwise(0.0).fill_nan(0).alias(f"w_{s}") for s in STATS]) \
        .filter(pl.col("e_neutral_pts").is_not_null())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player")
    ap.add_argument("--out", help="comma-separated teammates sitting")
    ap.add_argument("--date", type=date.fromisoformat, default=date.today())
    ap.add_argument("--tune", action="store_true", help="grid over K_PAIR on 2024-25")
    args = ap.parse_args()
    pg, rows, _, _, px, beta = _state()
    if args.player:
        name_id = dict(pg.select("player_name", "player_id").unique().iter_rows())
        out = {name_id[n.strip()] for n in (args.out or "").split(",") if n.strip()}
        res = tonight(name_id[args.player], out, args.date)
        print(f"{args.player}, out: {args.out}: multipliers", {s: round(v, 3) for s, v in res["mult"].items()})
        for m in res["mates"]:
            print(f"  without {m['mate']} ({m['games_without']} past games): " + "; ".join(
                f"{s} prior {m[s]['prior']:+.1%} own {('%+.1f%%' % (100 * m[s]['own'])) if m[s]['own'] is not None else '-'} -> {m[s]['effect']:+.1%}"
                for s in STATS))
        return
    print("prior beta (per 100 freed volume x own-rate tilt):", {s: {k: round(v, 3) for k, v in b.items()} for s, b in beta.items()})
    ks = (20.0, 40.0, 80.0, 160.0, 320.0, 1e9) if args.tune else (K_PAIR,)
    for k in ks:
        r = with_multipliers(rows, px, beta, k)
        for sid in (("22024",) if args.tune else ("22024", "22025", "22026")):
            print(f"K={k} season {sid}:", evaluate(r, sid))


if __name__ == "__main__":
    main()
