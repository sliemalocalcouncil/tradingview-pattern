"""Technical indicators + level clustering used by feature scoring."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# basic indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prior_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prior_close).abs(),
            (df["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).rolling(length, min_periods=length).mean()


def rvol(volume: pd.Series, length: int = 20) -> pd.Series:
    avg = volume.rolling(length, min_periods=length).mean()
    return volume / avg


def slope_pct(series: pd.Series, window: int = 21) -> pd.Series:
    """Linear-regression slope over `window` bars expressed as % per bar."""
    out = pd.Series(np.nan, index=series.index)
    if len(series) < window:
        return out
    x = np.arange(window)
    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1 : i + 1].values
        if np.isfinite(y).sum() < window:
            continue
        slope = np.polyfit(x, y, 1)[0]
        out.iloc[i] = slope / max(y[-1], 1e-9) * 100
    return out


def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)
    out["atr_14"] = atr(out, 14)
    out["atr_50"] = atr(out, 50)
    out["rvol_20"] = rvol(out["volume"], 20)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    return out


# ---------------------------------------------------------------------------
# pivot detection
# ---------------------------------------------------------------------------

@dataclass
class Pivot:
    idx: int
    timestamp: pd.Timestamp
    price: float
    kind: str  # "high" | "low"


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Pivot]:
    """Strict swing high/low pivots: a bar whose high (or low) is the local extremum
    within ±`left` bars on the left and ±`right` bars on the right.
    """
    if len(df) < left + right + 1:
        return []
    highs = df["high"].values
    lows = df["low"].values
    pivots: list[Pivot] = []
    for i in range(left, len(df) - right):
        h_window = highs[i - left : i + right + 1]
        if highs[i] == h_window.max() and (h_window == highs[i]).sum() == 1:
            pivots.append(Pivot(i, df.index[i], float(highs[i]), "high"))
        l_window = lows[i - left : i + right + 1]
        if lows[i] == l_window.min() and (l_window == lows[i]).sum() == 1:
            pivots.append(Pivot(i, df.index[i], float(lows[i]), "low"))
    return pivots


# ---------------------------------------------------------------------------
# level clustering (replaces single-pivot S/R from the old code; design §6.SR)
# ---------------------------------------------------------------------------

@dataclass
class Level:
    mid: float
    std: float
    touches: int
    last_touch_idx: int
    last_touch_ts: pd.Timestamp
    reaction_avg_atr: float  # avg reaction in ATR units


def find_levels(df: pd.DataFrame, lookback: int = 120, atr_bw_mult: float = 0.5,
                min_touches: int = 2, max_pivots: int = 30) -> list[Level]:
    """Cluster recent pivot highs+lows into 1D levels using ATR-bandwidth merging.

    Algorithm: simple greedy 1-D clustering by sorting pivot prices and grouping
    points within `atr_bw_mult * ATR(14)` of each other.
    """
    sub = df.tail(lookback)
    if len(sub) < 30 or "atr_14" not in sub.columns:
        return []
    a = sub["atr_14"].iloc[-1]
    if not np.isfinite(a) or a <= 0:
        return []
    bw = a * atr_bw_mult

    pivots = find_pivots(sub, left=3, right=3)
    if not pivots:
        return []
    pivots = pivots[-max_pivots:]
    pts = sorted(pivots, key=lambda p: p.price)

    clusters: list[list[Pivot]] = []
    current: list[Pivot] = [pts[0]]
    for p in pts[1:]:
        if abs(p.price - np.mean([q.price for q in current])) <= bw:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    levels: list[Level] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        prices = np.array([p.price for p in cluster])
        idxs = np.array([p.idx for p in cluster])
        ts_max = max(p.timestamp for p in cluster)
        # Reaction strength: how far did price travel away from the level after each touch (ATR units)
        reactions = []
        for p in cluster:
            j = p.idx
            if j + 5 >= len(sub):
                continue
            future = sub.iloc[j : j + 6]
            move = max(abs(future["high"].max() - p.price), abs(future["low"].min() - p.price))
            reactions.append(move / a)
        reaction_avg = float(np.mean(reactions)) if reactions else 0.0

        levels.append(
            Level(
                mid=float(np.mean(prices)),
                std=float(np.std(prices)),
                touches=int(len(cluster)),
                last_touch_idx=int(idxs.max()),
                last_touch_ts=ts_max,
                reaction_avg_atr=reaction_avg,
            )
        )
    # Sort by recency, descending
    levels.sort(key=lambda L: L.last_touch_idx, reverse=True)
    return levels


def nearest_levels(price: float, levels: list[Level]) -> tuple[Level | None, Level | None]:
    """Return (nearest_support, nearest_resistance) for a given price."""
    if not levels:
        return None, None
    above = [L for L in levels if L.mid > price]
    below = [L for L in levels if L.mid < price]
    res = min(above, key=lambda L: L.mid - price) if above else None
    sup = min(below, key=lambda L: price - L.mid) if below else None
    return sup, res


# ---------------------------------------------------------------------------
# helpers for normalization
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def lerp_score(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """Linearly map [lo, hi] -> [0, 100], saturating at the endpoints."""
    if hi == lo:
        return 100.0 if x >= hi else 0.0
    t = (x - lo) / (hi - lo)
    if invert:
        t = 1.0 - t
    return clamp(t * 100.0, 0.0, 100.0)


def bell_score(x: float, ideal_lo: float, ideal_hi: float, falloff: float = None) -> float:
    """100 inside [ideal_lo, ideal_hi]; falls off linearly to 0 over `falloff` distance."""
    if ideal_lo <= x <= ideal_hi:
        return 100.0
    falloff = falloff if falloff is not None else (ideal_hi - ideal_lo)
    if x < ideal_lo:
        return clamp((1 - (ideal_lo - x) / falloff) * 100.0, 0.0, 100.0)
    return clamp((1 - (x - ideal_hi) / falloff) * 100.0, 0.0, 100.0)
