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
PTS_UNIT = 2.0  # variance-to-mean of points (used when shrinking per-game rates)
# Share of points from free throws, twos and threes (league-typical). Drawing points as
# 1*Poisson + 2*Poisson + 3*Poisson keeps variance ~2.1x the mean AND allows odd totals;
# a single 2-point unit made every simulated total even, so P(15+) == P(16+).
PTS_MIX = ((1, 0.18), (2, 0.50), (3, 0.32))


def draw_points(rng, expected: np.ndarray) -> np.ndarray:
    """Points with the given expected values, as free throws + twos + threes."""
    return sum(v * rng.poisson(expected * share / v) for v, share in PTS_MIX)
# Backup-big minutes when the starter sits vs plays: 20.5 / 15.5 (docs/angles_abc_findings.md)
LEAGUE_UPLIFT_RATIO = 1.32
LEAGUE_UPLIFT_ADD = 5.0
HALF_LIFE: float | None = 5.0  # games; chosen on 2024-25, validated on 2025-26 (docs/context_model.md)
SHRINK_GAMES = 8  # pseudo-games of league-shifted minutes mixed into the starter-out sample
MINUTES_JITTER = 2.0


@dataclass
class PlayerFit:
    minutes: np.ndarray  # per past game
    starter_out: np.ndarray  # bool per past game
    pts_rate: np.ndarray  # per minute, shrunk; NaN where minutes < MIN_RATE_MINUTES
    reb_rate: np.ndarray
    ast_rate: np.ndarray
    weights: np.ndarray | None = None  # recency weight per template game (None = equal)

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


RECENCY_FLOOR = 0.2  # share of weight kept spread evenly over all history (protects the tails)


def recency_weights(n: int, half_life: float | None, floor: float | None = None) -> np.ndarray | None:
    """Weight per game, oldest first: halves every half_life games of age (None = equal).

    floor mixes in a uniform share so rare big games from further back still get sampled:
    w = (1 - floor) * decay / sum(decay) + floor / n.
    """
    if not half_life:
        return None
    phi = RECENCY_FLOOR if floor is None else floor
    decay = 0.5 ** (np.arange(n)[::-1] / half_life)
    return (1 - phi) * decay / decay.sum() + phi / n


def pick(rng, pool: np.ndarray, weights: np.ndarray | None, n: int) -> np.ndarray:
    """Draw n template indices from pool, weighted by recency if weights are given."""
    if weights is None:
        return rng.choice(pool, n)
    w = weights[pool]
    return rng.choice(pool, n, p=w / w.sum())


def fit_player(history: pl.DataFrame, half_life: float | None = None) -> PlayerFit | None:
    """history: this player's past games with minutes, pts, reb, ast, and optionally
    starter_out (treated as all False when absent, e.g. non-bigs)."""
    if history.height == 0:
        return None
    mins = history["minutes"].to_numpy().astype(float)
    rated = mins >= MIN_RATE_MINUTES
    if rated.sum() < 5:
        return None
    m = mins[rated]

    def rate(col: str, unit: float) -> np.ndarray:
        out = np.full(len(mins), np.nan)
        if col in history.columns:
            out[rated] = _shrink(history[col].to_numpy()[rated] / m, m, unit)
        else:
            out[rated] = 0.0
        return out

    starter_out = (
        history["starter_out"].to_numpy() if "starter_out" in history.columns
        else np.zeros(len(mins), bool)
    )
    hl = HALF_LIFE if half_life is None else half_life
    return PlayerFit(
        mins, starter_out, rate("pts", PTS_UNIT), rate("reb", 1.0), rate("ast", 1.0),
        recency_weights(len(mins), hl),
    )


def _templates(fit: PlayerFit, starter_out: bool, n: int, rng, uplift: str):
    """Pick a past game per simulation; return (template index, minutes)."""
    out_idx = np.flatnonzero(fit.starter_out)
    in_idx = np.flatnonzero(~fit.starter_out)
    if len(in_idx) == 0:
        in_idx = out_idx
    if not starter_out:
        idx = pick(rng, in_idx, fit.weights, n)
        return idx, fit.minutes[idx]
    w = len(out_idx) / (len(out_idx) + SHRINK_GAMES)
    use_own = (rng.random(n) < w) if len(out_idx) else np.zeros(n, bool)
    idx = np.where(use_own, pick(rng, out_idx if len(out_idx) else in_idx, fit.weights, n),
                   pick(rng, in_idx, fit.weights, n))
    base = fit.minutes[idx]
    scaled = base * LEAGUE_UPLIFT_RATIO if uplift == "ratio" else base + LEAGUE_UPLIFT_ADD
    return idx, np.where(use_own, base, scaled)


def simulate(fit: PlayerFit, starter_out: bool, n: int = 20_000, seed: int = 0,
             paired: bool = True, uplift: str = "ratio"):
    """Return simulated (minutes, pts, reb) arrays. See simulate_stats for assists too."""
    s = simulate_stats(fit, starter_out, n, seed, paired, uplift)
    return s["minutes"], s["pts"], s["reb"]


def simulate_stats(fit: PlayerFit, starter_out: bool, n: int = 20_000, seed: int = 0,
                   paired: bool = True, uplift: str = "ratio") -> dict[str, np.ndarray]:
    """Return simulated {"minutes", "pts", "reb", "ast"} arrays.

    paired=False draws the rates from an independent past game (ablation: ignores the
    minutes-production link). uplift: "ratio" or "add" for sparse starter-out history.
    """
    rng = np.random.default_rng(seed)
    idx, minutes = _templates(fit, starter_out, n, rng, uplift)
    minutes = np.clip(minutes + rng.normal(0, MINUTES_JITTER, n), 0, 48)
    rated = np.flatnonzero(~np.isnan(fit.pts_rate))
    # Short-minute templates have no reliable rate: borrow one from a random rated game.
    rate_idx = idx if paired else pick(rng, rated, fit.weights, n)
    missing = np.isnan(fit.pts_rate[rate_idx])
    rate_idx = np.where(missing, pick(rng, rated, fit.weights, n), rate_idx)
    pts_rate = np.clip(fit.pts_rate[rate_idx], 0, None)
    reb_rate = np.clip(fit.reb_rate[rate_idx], 0, None)
    ast_rate = np.clip(fit.ast_rate[rate_idx], 0, None)
    return {
        "minutes": minutes,
        "pts": draw_points(rng, pts_rate * minutes),
        "reb": rng.poisson(reb_rate * minutes),
        "ast": rng.poisson(ast_rate * minutes),
    }


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


STAT_CODES = {"p": "pts", "r": "reb", "a": "ast"}


def parse_combo(text: str) -> dict[str, int]:
    """'20p/10r/3a' -> {"pts": 20, "reb": 10, "ast": 3}."""
    out = {}
    for leg in text.lower().replace(" ", "").split("/"):
        out[STAT_CODES[leg[-1]]] = int(leg[:-1])
    return out


def combo_probs(stats: dict[str, np.ndarray], combos: list[dict[str, int]]) -> pl.DataFrame:
    """Joint P(every leg hits) vs the product of the legs' own probabilities."""
    rows = []
    for c in combos:
        legs = [stats[k] >= v for k, v in c.items()]
        joint = np.mean(np.logical_and.reduce(legs))
        indep = float(np.prod([leg.mean() for leg in legs]))
        rows.append({
            "combo": "/".join(f"{v}{k[0]}" for k, v in c.items()),
            "p": joint, "fair": 1 / joint if joint else float("inf"),
            "p_indep": indep, "fair_indep": 1 / indep if indep else float("inf"),
            "legs": ", ".join(f"{k}>={v}: {leg.mean():.2f}" for (k, v), leg in zip(c.items(), legs, strict=True)),
        })
    return pl.DataFrame(rows)
