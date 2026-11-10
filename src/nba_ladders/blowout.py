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


# Side-aware curves (10 Nov 2026), 2023-24..2025-26, 69k player-games: minutes vs prior 10, by |final margin| and
# whether his team won. The winners' bench (12-18 mpg, e.g. Drummond behind Towns) does NOT gain in a blowout win
# (+1.0 close -> +0.0-0.4 at 10-25); the losers' bench and the deep bench (<12) gain; starters lose on both sides.
MID_SIDE = np.array([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 35.0])
CURVE_SIDE = {
    ("starter", "W"): np.array([1.26, 0.94, -0.49, -2.34, -3.52, -4.80, -6.26]),
    ("starter", "L"): np.array([1.48, 0.87, -0.38, -1.82, -3.37, -4.47, -5.96]),
    ("rot24", "W"): np.array([0.92, 0.82, 0.12, -0.84, -1.36, -2.06, -3.61]),
    ("rot24", "L"): np.array([1.13, 0.92, -0.14, -0.78, -1.08, -2.12, -2.54]),
    ("rot18", "W"): np.array([0.71, 0.25, -0.02, 0.19, 0.05, 0.07, -1.06]),
    ("rot18", "L"): np.array([0.53, 0.44, 0.28, -0.11, 0.09, -0.32, 0.10]),
    ("bench", "W"): np.array([1.01, 1.09, 0.38, 0.13, 0.03, 0.61, 1.33]),
    ("bench", "L"): np.array([0.99, 1.03, 0.89, 0.85, 1.25, 1.83, 2.83]),
    ("deep", "W"): np.array([3.42, 2.89, 0.19, -0.11, -0.14, 1.39, 3.10]),
    ("deep", "L"): np.array([3.39, 3.02, 1.36, 1.74, 2.19, 4.23, 5.41]),
}


def role5(min20: float) -> str:
    return ("starter" if min20 >= 30 else "rot24" if min20 >= 24 else "rot18" if min20 >= 18
            else "bench" if min20 >= 12 else "deep")


def side_shift(margin: np.ndarray, min20: float) -> np.ndarray:
    """Minutes shift for each drawn final margin (his team's points minus theirs)."""
    r = role5(min20)
    return np.where(margin > 0, np.interp(np.abs(margin), MID_SIDE, CURVE_SIDE[(r, "W")]),
                    np.interp(np.abs(margin), MID_SIDE, CURVE_SIDE[(r, "L")]))


def side_base(min20: float, n: int = 40_000) -> float:
    """The same shift averaged over a typical game (±4.5 spread, either side): his average already contains these."""
    m = np.random.default_rng(1).normal(0, 1, n) * SD_MARGIN + np.where(np.arange(n) % 2, TYPICAL_SPREAD, -TYPICAL_SPREAD)
    return float(side_shift(m, min20).mean())
