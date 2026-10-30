"""Calibrated fair probability: Sportsbet's price + the model, weighted as the backtest learned.

    uv run python -m nba_ladders.blend fit      # fit on market_backtest.parquet, save params

logit(p) = w . [1, lb, lm] + per-stat (reb, ast, pts+reb) [1, lb, lm] + assist/rebound luck
(>= 1.10) [1, lm] + starter-out [1, lm], where lb = logit(1 / Sportsbet price) and
lm = logit(model probability). Fit by penalised logistic regression on two seasons of
historical Sportsbet rungs (664k). Raw model-vs-book disagreements lose money in the
backtest; this blend is what the live sheet uses as its headline fair price.
"""

import json
import sys

import numpy as np
import polars as pl
from scipy.optimize import minimize

from nba_ladders import config

PARAMS = config.ROOT / "src" / "nba_ladders" / "blend_params.json"
STATS = ["reb", "ast", "pts+reb"]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(p_book, p_model, stat, luck_hi, starter_out) -> np.ndarray:
    lb, lm = _logit(p_book), _logit(p_model)
    stat = np.asarray(stat)
    cols = [np.ones_like(lb), lb, lm]
    for s in STATS:
        d = (stat == s).astype(float)
        cols += [d, d * lb, d * lm]
    for flag in (np.asarray(luck_hi, float), np.asarray(starter_out, float)):
        cols += [flag, flag * lm]
    return np.column_stack(cols)


def fit(df: pl.DataFrame) -> list[float]:
    luck_stat = np.where(df["stat"].to_numpy() == "ast", df["luck_ast"].to_numpy(), df["luck_reb"].to_numpy())
    x = design(df["p_book"].to_numpy(), df["p_model"].to_numpy(), df["stat"].to_numpy(),
               luck_stat >= 1.10, df["starter_out"].to_numpy())
    y = df["hit"].to_numpy().astype(float)

    def nll(w):
        z = x @ w
        return np.mean(np.logaddexp(0, z) - y * z) + 1e-4 * np.sum(w[1:] ** 2)

    return minimize(nll, np.zeros(x.shape[1]), method="BFGS").x.tolist()


def predict(p_book, p_model, stat, luck_hi, starter_out, w=None) -> np.ndarray:
    w = np.asarray(w if w is not None else json.loads(PARAMS.read_text())["w"])
    z = design(np.atleast_1d(p_book), np.atleast_1d(p_model), np.atleast_1d(stat),
               np.atleast_1d(luck_hi), np.atleast_1d(starter_out)) @ w
    return 1 / (1 + np.exp(-z))


def main() -> None:
    if sys.argv[1:] != ["fit"]:
        raise SystemExit("usage: python -m nba_ladders.blend fit")
    df = pl.read_parquet(config.DATA_PROCESSED / "market_backtest.parquet")
    # out-of-sample check first: fit on 2024-25, score on 2025-26
    from datetime import date

    tr, te = df.filter(pl.col("game_date") < date(2025, 8, 1)), df.filter(pl.col("game_date") >= date(2025, 8, 1))
    w = fit(tr)
    luck = np.where(te["stat"].to_numpy() == "ast", te["luck_ast"].to_numpy(), te["luck_reb"].to_numpy())
    p = predict(te["p_book"].to_numpy(), te["p_model"].to_numpy(), te["stat"].to_numpy(), luck >= 1.10,
                te["starter_out"].to_numpy(), w)
    y = te["hit"].to_numpy().astype(float)
    ll = lambda q: float(-np.mean(y * np.log(np.clip(q, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-4, 1))))
    print(f"fit 2024-25 -> 2025-26: log loss book {ll(te['p_book'].to_numpy()):.5f}, blend {ll(p):.5f}")
    w_all = fit(df)
    PARAMS.write_text(json.dumps({"w": w_all, "n": df.height, "fit_on": "market_backtest.parquet"}))
    print(f"saved {PARAMS} (fit on all {df.height:,} rungs)")


if __name__ == "__main__":
    main()
