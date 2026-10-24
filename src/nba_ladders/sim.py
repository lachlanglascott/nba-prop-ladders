"""Monte Carlo simulator for a player's joint points/rebounds in one game.

Each simulated game resamples one of the player's past games as a template:
    minutes  = that game's minutes (+ smoothing). For a starter-out game with sparse starter-out
               history, the template may be a starter-in game scaled up by the league uplift
               (angle C: the uplift is situational, not a player trait)
    rates    = that same game's (pts/min, reb/min), so a game where he was playing well carries
               both its extra minutes and its production ("playing well -> stays on court"),
               and pts/reb keep their within-game correlation. Rates are shrunk toward his mean
               to strip the sampling noise that the count step adds back.
    counts   ~ Poisson at the simulated minutes (points in 2-point units)

Only history strictly before the game is used, so backtests are as-at.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

RUNGS = [(8, 8), (10, 10), (12, 10), (12, 12), (15, 10), (15, 12), (15, 15), (20, 15)]
MIN_RATE_MINUTES = 10  # games shorter than this give too-noisy per-minute rates
PTS_UNIT = 2.0
# Backup-big minutes when the starter sits vs plays: 20.5 / 15.5 (docs/angles_abc_findings.md)
LEAGUE_UPLIFT_RATIO = 1.32
LEAGUE_UPLIFT_ADD = 5.0
SHRINK_GAMES = 8  # pseudo-games of league-shifted minutes mixed into the starter-out sample
MINUTES_JITTER = 2.0


@dataclass
class PlayerFit:
    minutes: np.ndarray  # per past game
    starter_out: np.ndarray  # bool per past game
    pts_rate: np.ndarray  # per minute, shrunk; NaN where minutes < MIN_RATE_MINUTES
    reb_rate: np.ndarray

    @property
    def n_games(self) -> int:
        return len(self.minutes)


def _shrink(rates: np.ndarray, minutes: np.ndarray, unit: float) -> np.ndarray:
    """Pull per-game rates toward the mean so their spread is true game-to-game variation only.

    Observed var(rate) = true var + Poisson noise (≈ unit * mean / minutes, averaged).
    """
    mean = rates.mean()
    total = rates.var()
    noise = (unit * mean / minutes).mean()
    k = np.sqrt(max(total - noise, 0.0) / total) if total > 0 else 0.0
    return mean + k * (rates - mean)


def fit_player(history: pl.DataFrame) -> PlayerFit | None:
    """history: this player's past games with minutes, pts, reb, starter_out."""
    if history.height == 0:
        return None
    mins = history["minutes"].to_numpy().astype(float)
    rated = mins >= MIN_RATE_MINUTES
    if rated.sum() < 5:
        return None
    pts_rate = np.full(len(mins), np.nan)
    reb_rate = np.full(len(mins), np.nan)
    m = mins[rated]
    pts_rate[rated] = _shrink(history["pts"].to_numpy()[rated] / m, m, PTS_UNIT)
    reb_rate[rated] = _shrink(history["reb"].to_numpy()[rated] / m, m, 1.0)
    return PlayerFit(mins, history["starter_out"].to_numpy(), pts_rate, reb_rate)


def _templates(fit: PlayerFit, starter_out: bool, n: int, rng, uplift: str):
    """Pick a past game per simulation; return (template index, minutes)."""
    out_idx = np.flatnonzero(fit.starter_out)
    in_idx = np.flatnonzero(~fit.starter_out)
    if len(in_idx) == 0:
        in_idx = out_idx
    if not starter_out:
        idx = rng.choice(in_idx, n)
        return idx, fit.minutes[idx]
    w = len(out_idx) / (len(out_idx) + SHRINK_GAMES)
    use_own = (rng.random(n) < w) if len(out_idx) else np.zeros(n, bool)
    idx = np.where(use_own, rng.choice(out_idx if len(out_idx) else in_idx, n),
                   rng.choice(in_idx, n))
    base = fit.minutes[idx]
    scaled = base * LEAGUE_UPLIFT_RATIO if uplift == "ratio" else base + LEAGUE_UPLIFT_ADD
    return idx, np.where(use_own, base, scaled)


def simulate(fit: PlayerFit, starter_out: bool, n: int = 20_000, seed: int = 0,
             paired: bool = True, uplift: str = "ratio"):
    """Return simulated (minutes, pts, reb) arrays.

    paired=False draws the rate from an independent past game (ablation: ignores the
    minutes-production link). uplift: "ratio" or "add" for sparse starter-out history.
    """
    rng = np.random.default_rng(seed)
    idx, minutes = _templates(fit, starter_out, n, rng, uplift)
    minutes = np.clip(minutes + rng.normal(0, MINUTES_JITTER, n), 0, 48)
    rated = np.flatnonzero(~np.isnan(fit.pts_rate))
    # Short-minute templates have no reliable rate: borrow one from a random rated game.
    rate_idx = idx if paired else rng.choice(rated, n)
    missing = np.isnan(fit.pts_rate[rate_idx])
    rate_idx = np.where(missing, rng.choice(rated, n), rate_idx)
    pts_rate = np.clip(fit.pts_rate[rate_idx], 0, None)
    reb_rate = np.clip(fit.reb_rate[rate_idx], 0, None)
    pts = PTS_UNIT * rng.poisson(pts_rate * minutes / PTS_UNIT)
    reb = rng.poisson(reb_rate * minutes)
    return minutes, pts, reb


def rung_probs(pts: np.ndarray, reb: np.ndarray, rungs=RUNGS) -> dict:
    """Joint P(pts >= a and reb >= b), and the same if the legs were independent."""
    out = {}
    for a, b in rungs:
        joint = np.mean((pts >= a) & (reb >= b))
        indep = np.mean(pts >= a) * np.mean(reb >= b)
        out[f"{a}/{b}"] = {"joint": joint, "indep": indep}
    return out


def price_table(pts: np.ndarray, reb: np.ndarray, rungs=RUNGS) -> pl.DataFrame:
    rows = [
        {"rung": k, "p": v["joint"], "fair": 1 / v["joint"] if v["joint"] else float("inf"),
         "p_indep": v["indep"], "fair_indep": 1 / v["indep"] if v["indep"] else float("inf")}
        for k, v in rung_probs(pts, reb, rungs).items()
    ]
    return pl.DataFrame(rows)
