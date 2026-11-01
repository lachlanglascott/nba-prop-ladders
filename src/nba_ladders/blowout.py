"""Expected minutes shift from the point spread (blowout risk).

Measured on 2023-24..2025-26: minutes vs each player's prior 10-game average, by final margin.
Starters (30+ mpg) lose ~2-6 minutes once a game is decided by 15+ and gain ~1 in close games;
20-30 mpg players lose less; the deep bench gains in blowouts. Final margin ~ Normal(spread, 12.5),
so the expected shift for tonight is E[curve(|margin|)] at tonight's spread minus at a typical
spread (4.5), since each player's average already contains ordinary games.
"""

import numpy as np

SD_MARGIN = 12.5
TYPICAL_SPREAD = 4.5
MID = np.array([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 35.0])
CURVE = {  # minutes vs prior 10-game average, by final margin bucket
    "starter": np.array([1.27, 0.87, -0.52, -2.25, -3.61, -4.70, -6.17]),
    "rotation": np.array([0.69, 0.48, -0.16, -0.83, -1.00, -1.56, -2.31]),
    "bench": np.array([1.37, 1.26, 0.25, 0.35, 0.39, 1.57, 2.50]),
}


def role(min20: float) -> str:
    return "starter" if min20 >= 30 else "rotation" if min20 >= 20 else "bench"


def _expected(spread: float, curve: np.ndarray) -> float:
    m = np.abs(np.random.default_rng(0).normal(spread, SD_MARGIN, 20_000))
    return float(np.interp(m, MID, curve).mean())


def minutes_shift(spread: float | None, min20: float) -> float:
    """Expected minutes change tonight for a player averaging min20, given his team's spread."""
    if spread is None:
        return 0.0
    c = CURVE[role(min20)]
    return _expected(abs(spread), c) - _expected(TYPICAL_SPREAD, c)
