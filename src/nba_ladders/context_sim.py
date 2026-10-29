"""Context-aware simulator: any combination of absent teammates.

Each past game is a template. Its minutes and per-minute rates are first *un-adjusted* for the
teammates who sat that night (back to a full-strength baseline), then *re-adjusted* for the
target game's absences:

    minutes = template_min - shift(that night) + shift(target)
    rate    = template_rate / mult(that night) * mult(target)

Effects come from absences.py (pair with/without, shrunk to a league prior). Counts are drawn
as in sim.py (points as free throws + twos + threes), keeping minutes, PPM, RPM and APM from the same template game.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from nba_ladders import sim
from nba_ladders.sim import (
    MIN_RATE_MINUTES,
    MINUTES_JITTER,
    PTS_UNIT,
    _shrink,
    draw_points,
    pick,
    recency_weights,
)

RATES = ["pts", "reb", "ast"]
UNITS = {"pts": PTS_UNIT, "reb": 1.0, "ast": 1.0}


@dataclass
class ContextFit:
    base_minutes: np.ndarray  # per template game, un-adjusted to full strength
    base_rates: dict[str, np.ndarray]  # per template game, NaN where < MIN_RATE_MINUTES
    weights: np.ndarray | None = None  # recency weight per template game
    role_shift: float = 0.0  # current minutes level vs history (sim.role_shift)


def fit_context(history: pl.DataFrame, half_life: float | None = None) -> ContextFit | None:
    """history: player's past games with minutes, pts, reb, ast, shift_min, mult_<stat>."""
    if history.height == 0:
        return None
    mins = history["minutes"].to_numpy().astype(float)
    rated = mins >= MIN_RATE_MINUTES
    if rated.sum() < 5:
        return None
    base_min = np.clip(mins - history["shift_min"].to_numpy(), 0, 48)
    m = mins[rated]
    rates = {}
    for s in RATES:
        r = np.full(len(mins), np.nan)
        raw = history[s].to_numpy()[rated] / m
        r[rated] = _shrink(raw, m, UNITS[s]) / history[f"mult_{s}"].to_numpy()[rated]
        rates[s] = r
    hl = sim.HALF_LIFE if half_life is None else half_life
    weights = recency_weights(len(mins), hl)
    return ContextFit(base_min, rates, weights, sim.role_shift(base_min, weights))


def simulate_context(fit: ContextFit, shift_min: float = 0.0, mult: dict | None = None,
                     n: int = 20_000, seed: int = 0) -> dict[str, np.ndarray]:
    """Simulate a game with the given absence context (shift, rate multipliers)."""
    mult = mult or {}
    rng = np.random.default_rng(seed)
    idx = pick(rng, np.arange(len(fit.base_minutes)), fit.weights, n)
    minutes = np.clip(fit.base_minutes[idx] + fit.role_shift + shift_min
                      + rng.normal(0, MINUTES_JITTER, n), 0, 48)
    rated = np.flatnonzero(~np.isnan(fit.base_rates["pts"]))
    rate_idx = np.where(np.isnan(fit.base_rates["pts"][idx]), pick(rng, rated, fit.weights, n), idx)
    out = {"minutes": minutes}
    for s in RATES:
        rate = np.clip(fit.base_rates[s][rate_idx] * mult.get(s, 1.0), 0, None)
        out[s] = draw_points(rng, rate * minutes) if s == "pts" else rng.poisson(rate * minutes)
    return out
