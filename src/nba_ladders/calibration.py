"""Recalibrate simulated probabilities: logit(p') = a + b * logit(p).

The simulator's raw probabilities run slightly low on long shots and on point thresholds
(docs/context_model.md). A two-parameter logistic recalibration, fit on one season's backtest
and checked on the next, fixes the overall level (a) and the spread (b) without touching the
ordering of bets.

    uv run python -m nba_ladders.calibration fit.parquet check.parquet
"""

import json
import sys

import numpy as np
import polars as pl
from scipy.optimize import minimize

from nba_ladders import config

PARAMS_PATH = config.ROOT / "src" / "nba_ladders" / "calibration_params.json"
EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit(p: np.ndarray, hit: np.ndarray) -> tuple[float, float]:
    x = _logit(p)

    def nll(theta):
        z = theta[0] + theta[1] * x
        return np.mean(np.logaddexp(0, z) - hit * z)

    res = minimize(nll, x0=[0.0, 1.0], method="BFGS")
    return float(res.x[0]), float(res.x[1])


def apply(p, a: float, b: float):
    return 1 / (1 + np.exp(-(a + b * _logit(np.asarray(p, dtype=float)))))


def load_params() -> tuple[float, float]:
    if PARAMS_PATH.exists():
        d = json.loads(PARAMS_PATH.read_text())
        return d["a"], d["b"]
    return 0.0, 1.0


def log_loss(p: np.ndarray, hit: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(hit * np.log(p) + (1 - hit) * np.log(1 - p)))


def report(df: pl.DataFrame, a: float, b: float, col: str = "p_ctx") -> pl.DataFrame:
    p, hit = df[col].to_numpy(), df["hit"].to_numpy().astype(float)
    q = apply(p, a, b)
    bands = [("p<10% (fair >$10)", p < 0.10), ("10-30%", (p >= 0.10) & (p < 0.30)),
             ("30%+", p >= 0.30), ("all", np.ones_like(p, bool))]
    return pl.DataFrame([
        {"band": name, "n": int(m.sum()), "hits": int(hit[m].sum()),
         "exp_raw": round(float(p[m].sum())), "exp_cal": round(float(q[m].sum())),
         "ll_raw": round(log_loss(p[m], hit[m]), 5), "ll_cal": round(log_loss(q[m], hit[m]), 5)}
        for name, m in bands
    ])


def main() -> None:
    fit_path, check_path = sys.argv[1], sys.argv[2]
    tr = pl.read_parquet(fit_path)
    a, b = fit(tr["p_ctx"].to_numpy(), tr["hit"].to_numpy().astype(float))
    print(f"fit on {fit_path}: a={a:.4f}, b={b:.4f}")
    with pl.Config(tbl_hide_column_data_types=True):
        print("fit season:"); print(report(tr, a, b))
        print("check season:"); print(report(pl.read_parquet(check_path), a, b))
    if "--save" in sys.argv:
        PARAMS_PATH.write_text(json.dumps({"a": a, "b": b, "fit_on": fit_path}))
        print(f"saved {PARAMS_PATH}")


if __name__ == "__main__":
    main()
