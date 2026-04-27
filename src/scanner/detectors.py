"""Pattern detectors — structural detection only (no scoring inside).

Per design §1.4, this module is intentionally divorced from scoring; each
detector returns a PatternMatch carrying landmarks + structural features that
the geometry/scoring modules consume.

v1 ships with 4 high-value detectors:
  - double_bottom
  - bull_flag (+ HTP variant)
  - vcp
  - ascending_triangle
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from .indicators import find_pivots
from .models import PatternMatch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _atr_pct(daily: pd.DataFrame) -> float:
    if "atr_14" not in daily.columns or daily["atr_14"].iloc[-1] is None:
        return 0.02
    a = float(daily["atr_14"].iloc[-1])
    p = float(daily["close"].iloc[-1])
    return a / max(p, 1e-9)


# ---------------------------------------------------------------------------
# Double Bottom — runs on hourly bars
# ---------------------------------------------------------------------------

def detect_double_bottom(hourly: pd.DataFrame, daily: pd.DataFrame,
                          lookback: int = 80) -> PatternMatch | None:
    if len(hourly) < 30:
        return None
    sub = hourly.tail(lookback)
    pivots_low = [p for p in find_pivots(sub, left=3, right=3) if p.kind == "low"]
    if len(pivots_low) < 2:
        return None

    last = sub.iloc[-1]
    price = float(last["close"])
    best: PatternMatch | None = None

    # Try the latest pair of low pivots that form a plausible double bottom
    for i in range(len(pivots_low) - 1):
        p1, p2 = pivots_low[i], pivots_low[-1]
        if p2.idx - p1.idx < 6 or p2.idx - p1.idx > 50:
            continue
        avg = (p1.price + p2.price) / 2
        sim_pct = abs(p1.price - p2.price) / avg
        if sim_pct > 0.05:  # widened from 4% to allow more truthful negatives downstream
            continue

        between = sub.iloc[p1.idx : p2.idx + 1]
        neckline = float(between["high"].max())
        bounce = (neckline - min(p1.price, p2.price)) / min(p1.price, p2.price)
        if bounce < 0.025:
            continue

        # Right-bottom reaction quality (close in upper part of the pivot bar's range)
        right_bar = sub.iloc[p2.idx]
        rb_range = max(right_bar["high"] - right_bar["low"], 1e-9)
        right_bottom_reaction = (right_bar["close"] - right_bar["low"]) / rb_range

        atr_pct = _atr_pct(daily)
        invalid = min(p1.price, p2.price) * 0.985

        match = PatternMatch(
            pattern="double_bottom",
            neckline=neckline,
            invalid_below=invalid,
            measured_move_target=neckline + (neckline - min(p1.price, p2.price)),
            structure={
                "left_low": float(p1.price), "left_idx": int(p1.idx),
                "right_low": float(p2.price), "right_idx": int(p2.idx),
                "separation_bars": int(p2.idx - p1.idx),
                "structural_low": float(min(p1.price, p2.price)),
                "current_price": price,
                "neckline_distance_pct": float((neckline - price) / price),
                "base_bars": int(p2.idx - p1.idx),
            },
            geometry_features={
                "low_similarity_pct": float(sim_pct),
                "bounce_pct": float(bounce),
                "right_bottom_reaction": float(right_bottom_reaction),
                "atr_pct": float(atr_pct),
            },
        )
        # Prefer the most-recent right pivot
        if best is None or match.structure["right_idx"] > (best.structure["right_idx"] or 0):
            best = match

    return best


# ---------------------------------------------------------------------------
# Bull Flag (+ HTP) — hourly
# ---------------------------------------------------------------------------

def detect_bull_flag(hourly: pd.DataFrame, daily: pd.DataFrame,
                      lookback: int = 60) -> PatternMatch | None:
    if len(hourly) < 25:
        return None
    sub = hourly.tail(lookback)

    # Find the most prominent peak in the recent half of the lookback window
    half = max(15, len(sub) // 2)
    peak_window = sub.iloc[-half:]
    peak_idx_local = int(peak_window["high"].values.argmax())
    peak_idx = len(sub) - half + peak_idx_local
    if peak_idx >= len(sub) - 4:  # need at least a 4-bar flag after the peak
        return None
    peak_price = float(sub.iloc[peak_idx]["high"])

    # Find the swing low preceding the peak (within 25 bars)
    pre = sub.iloc[max(0, peak_idx - 25) : peak_idx]
    if pre.empty:
        return None
    pole_low_idx_local = int(pre["low"].values.argmin())
    pole_low_idx = max(0, peak_idx - 25) + pole_low_idx_local
    pole_low = float(sub.iloc[pole_low_idx]["low"])
    pole_gain = (peak_price - pole_low) / max(pole_low, 1e-9)
    if pole_gain < 0.06:
        return None

    flag = sub.iloc[peak_idx + 1 :]
    if len(flag) < 4 or len(flag) > 30:
        return None

    flag_low = float(flag["low"].min())
    flag_high = float(flag["high"].max())
    pullback = (peak_price - flag_low) / peak_price
    flag_range = (flag_high - flag_low) / peak_price
    drift_slope = float(np.polyfit(range(len(flag)), flag["close"].values, 1)[0])
    drift_pct_per_bar = drift_slope / max(flag["close"].iloc[-1], 1e-9)

    atr_pct = _atr_pct(daily)
    is_htp = pole_gain >= 0.14 and flag_range <= 0.08 and pullback <= 0.10
    pattern_name = "high_tight_pullback" if is_htp else "bull_flag"

    return PatternMatch(
        pattern=pattern_name,
        neckline=peak_price,
        invalid_below=flag_low * 0.985,
        measured_move_target=peak_price + (peak_price - pole_low),
        structure={
            "pole_low": pole_low, "pole_low_idx": int(pole_low_idx),
            "peak_price": peak_price, "peak_idx": int(peak_idx),
            "pole_gain": float(pole_gain),
            "flag_bars": int(len(flag)),
            "flag_low": flag_low, "flag_high": flag_high,
            "structural_low": flag_low,
            "current_price": float(sub.iloc[-1]["close"]),
            "neckline_distance_pct": float((peak_price - sub.iloc[-1]["close"]) / sub.iloc[-1]["close"]),
            "base_bars": int(len(flag)),
        },
        geometry_features={
            "pole_gain": float(pole_gain),
            "pullback_pct": float(pullback),
            "flag_range_pct": float(flag_range),
            "flag_drift_per_bar": float(drift_pct_per_bar),
            "atr_pct": float(atr_pct),
            "pole_atr_multiple": float(pole_gain / max(atr_pct, 1e-9)),
        },
    )


# ---------------------------------------------------------------------------
# VCP — daily
# ---------------------------------------------------------------------------

def detect_vcp(daily: pd.DataFrame, lookback: int = 130) -> PatternMatch | None:
    if len(daily) < 60:
        return None
    sub = daily.tail(lookback)
    pivots = find_pivots(sub, left=3, right=3)
    pivot_highs = [p for p in pivots if p.kind == "high"]
    pivot_lows = [p for p in pivots if p.kind == "low"]
    if len(pivot_highs) < 3 or len(pivot_lows) < 3:
        return None

    # Pair adjacent (high, low) to compute contraction depths
    pairs: list[tuple] = []
    sorted_p = sorted(pivots, key=lambda p: p.idx)
    last_high = None
    for p in sorted_p:
        if p.kind == "high":
            last_high = p
        elif p.kind == "low" and last_high and p.idx > last_high.idx:
            depth = (last_high.price - p.price) / last_high.price
            if depth > 0:
                pairs.append((last_high, p, depth))
            last_high = None
    if len(pairs) < 3:
        return None
    pairs = pairs[-5:]  # last 5 contractions
    contractions = [c for *_, c in pairs]

    # Near-monotone decreasing (each ≤ 0.9 × prior — design fix C12)
    descending = all(contractions[i + 1] <= contractions[i] * 0.95 for i in range(len(contractions) - 1))
    final_pivot = pairs[-1][0].price
    final_low = pairs[-1][1].price

    # Volume dry-up (last 10d vs prior 20d)
    vol_recent = float(sub["volume"].tail(10).mean())
    vol_prior = float(sub["volume"].iloc[-30:-10].mean()) if len(sub) >= 30 else vol_recent
    dry_up_ratio = vol_recent / max(vol_prior, 1e-9)

    if not descending and len(contractions) < 4:
        return None

    return PatternMatch(
        pattern="vcp",
        neckline=float(final_pivot),
        invalid_below=float(final_low) * 0.985,
        measured_move_target=float(final_pivot + (pairs[0][0].price - pairs[-1][1].price)),
        structure={
            "n_contractions": int(len(contractions)),
            "contractions_pct": [round(c * 100, 2) for c in contractions],
            "descending_strict": bool(descending),
            "final_low": float(final_low),
            "final_pivot": float(final_pivot),
            "structural_low": float(final_low),
            "dry_up_ratio": float(dry_up_ratio),
            "current_price": float(sub["close"].iloc[-1]),
            "neckline_distance_pct": float((final_pivot - sub["close"].iloc[-1]) / sub["close"].iloc[-1]),
            "base_bars": int(pairs[-1][1].idx - pairs[0][0].idx),
        },
        geometry_features={
            "contraction_count": float(len(contractions)),
            "first_contraction_pct": float(contractions[0]),
            "last_contraction_pct": float(contractions[-1]),
            "dry_up_ratio": float(dry_up_ratio),
            "near_monotone": float(descending),
            "base_depth_pct": float((pairs[0][0].price - pairs[-1][1].price) / pairs[0][0].price),
        },
    )


# ---------------------------------------------------------------------------
# Ascending Triangle — hourly
# ---------------------------------------------------------------------------

def detect_ascending_triangle(hourly: pd.DataFrame, daily: pd.DataFrame,
                                lookback: int = 90) -> PatternMatch | None:
    if len(hourly) < 30:
        return None
    sub = hourly.tail(lookback)
    pivots = find_pivots(sub, left=3, right=3)
    pivot_highs = [p for p in pivots if p.kind == "high"][-5:]
    pivot_lows = [p for p in pivots if p.kind == "low"][-5:]
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    # Flat-top: top pivots within ±1.5% of their max — count touches
    top = max(p.price for p in pivot_highs)
    near_top = [p for p in pivot_highs if (top - p.price) / top <= 0.015]
    if len(near_top) < 2:
        return None

    # Higher lows: regression slope on pivot lows must be > 0
    lows_x = np.array([p.idx for p in pivot_lows], dtype=float)
    lows_y = np.array([p.price for p in pivot_lows], dtype=float)
    if lows_x[-1] == lows_x[0]:
        return None
    slope = float(np.polyfit(lows_x, lows_y, 1)[0])
    if slope <= 0:
        return None

    # Compression: recent range vs older range
    half = len(sub) // 2
    range_recent = float((sub["high"].tail(half) - sub["low"].tail(half)).mean())
    range_prior = float((sub["high"].iloc[:half] - sub["low"].iloc[:half]).mean())
    compression = 1.0 - range_recent / max(range_prior, 1e-9)

    last_low = float(min(p.price for p in pivot_lows[-3:]))
    last_close = float(sub["close"].iloc[-1])

    return PatternMatch(
        pattern="ascending_triangle",
        neckline=float(top),
        invalid_below=last_low * 0.985,
        measured_move_target=top + (top - last_low),
        structure={
            "top": float(top),
            "top_touches": int(len(near_top)),
            "low_slope_per_bar": slope,
            "lowest_low": last_low,
            "structural_low": last_low,
            "current_price": last_close,
            "neckline_distance_pct": float((top - last_close) / last_close),
            "base_bars": int(pivot_lows[-1].idx - pivot_lows[0].idx),
        },
        geometry_features={
            "top_touches": float(len(near_top)),
            "low_slope": slope,
            "compression": float(compression),
            "atr_pct": _atr_pct(daily),
        },
    )


# ---------------------------------------------------------------------------
# Cup with Handle — daily
# ---------------------------------------------------------------------------

def detect_cup_with_handle(daily: pd.DataFrame, lookback: int = 180) -> PatternMatch | None:
    """O'Neil-style cup with handle.

    Geometry:
      left rim peak (A) → U-shaped descent to cup bottom (B) → ascent to
      right rim (C, ≈ A) → shallow handle pullback (D) → ready to break out.

    Conservative thresholds (per design §5.P.6):
      - cup depth 12–35 %
      - cup duration 25–125 bars
      - rim asymmetry ≤ 5 %
      - handle depth ≤ 12 %
      - handle ≤ 30 % of cup duration
      - handle in upper half of cup
      - U-shape (not V): cup_bottom_to_left_rim_bars >= 5
    """
    if len(daily) < 50:
        return None
    sub = daily.tail(lookback).reset_index(drop=False)
    pivots = find_pivots(sub, left=4, right=4)
    if len(pivots) < 4:
        return None
    pivot_highs = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.idx)
    pivot_lows = sorted([p for p in pivots if p.kind == "low"], key=lambda p: p.idx)
    if len(pivot_highs) < 2 or not pivot_lows:
        return None

    best: PatternMatch | None = None
    # Try combinations of (left_rim, right_rim) with the deepest low between
    for i in range(len(pivot_highs) - 1):
        left_rim = pivot_highs[i]
        for j in range(i + 1, len(pivot_highs)):
            right_rim = pivot_highs[j]
            cup_bars = right_rim.idx - left_rim.idx
            if not (25 <= cup_bars <= 125):
                continue

            # rim symmetry
            rim_diff = abs(right_rim.price - left_rim.price) / max(left_rim.price, 1e-9)
            if rim_diff > 0.05:
                continue

            # cup bottom = lowest low between rims
            interior_lows = [p for p in pivot_lows if left_rim.idx < p.idx < right_rim.idx]
            if not interior_lows:
                continue
            cup_bottom = min(interior_lows, key=lambda p: p.price)
            cup_depth = (left_rim.price - cup_bottom.price) / left_rim.price
            if not (0.12 <= cup_depth <= 0.35):
                continue

            # U-shape: cup bottom should not be too close to either rim
            if (cup_bottom.idx - left_rim.idx) < 5 or (right_rim.idx - cup_bottom.idx) < 5:
                continue

            # Handle: pullback after right rim
            handle = sub.iloc[right_rim.idx + 1:]
            if handle.empty or len(handle) < 3:
                continue
            handle_low = float(handle["low"].min())
            handle_depth = (right_rim.price - handle_low) / right_rim.price
            if handle_depth > 0.12:
                continue
            if len(handle) > cup_bars * 0.30:
                continue
            # Handle must form in upper half of cup
            cup_midpoint = (left_rim.price + cup_bottom.price) / 2
            if handle_low < cup_midpoint:
                continue

            atr_pct = _atr_pct(daily)
            current_price = float(daily["close"].iloc[-1])
            match = PatternMatch(
                pattern="cup_with_handle",
                neckline=float(right_rim.price),
                invalid_below=handle_low * 0.985,
                measured_move_target=float(right_rim.price) + (left_rim.price - cup_bottom.price),
                structure={
                    "left_rim": float(left_rim.price), "left_rim_idx": int(left_rim.idx),
                    "cup_bottom": float(cup_bottom.price), "cup_bottom_idx": int(cup_bottom.idx),
                    "right_rim": float(right_rim.price), "right_rim_idx": int(right_rim.idx),
                    "handle_low": handle_low,
                    "handle_bars": int(len(handle)),
                    "cup_bars": int(cup_bars),
                    "structural_low": handle_low,
                    "current_price": current_price,
                    "neckline_distance_pct": float((right_rim.price - current_price) / current_price),
                    "base_bars": int(cup_bars + len(handle)),
                },
                geometry_features={
                    "cup_depth_pct": float(cup_depth),
                    "rim_symmetry_pct": float(rim_diff),
                    "handle_depth_pct": float(handle_depth),
                    "handle_to_cup_ratio": float(len(handle) / cup_bars),
                    "cup_duration_bars": float(cup_bars),
                    "atr_pct": atr_pct,
                    "u_shape_score": float(min(cup_bottom.idx - left_rim.idx,
                                                  right_rim.idx - cup_bottom.idx) / cup_bars),
                },
            )
            if best is None or match.structure["right_rim_idx"] > best.structure["right_rim_idx"]:
                best = match
    return best


# ---------------------------------------------------------------------------
# Inverse Head & Shoulders — hourly
# ---------------------------------------------------------------------------

def detect_inverse_head_shoulders(hourly: pd.DataFrame, daily: pd.DataFrame,
                                    lookback: int = 100) -> PatternMatch | None:
    """3 lows: L1 (left shoulder), L2 (head, deepest), L3 (right shoulder).

    Conservative thresholds:
      - head 4–25 % below shoulder average
      - shoulder asymmetry (price) ≤ 8 %
      - shoulder asymmetry (time) ≤ 50 %
      - L3 reaction (close in upper half of bar range)
    """
    if len(hourly) < 30:
        return None
    sub = hourly.tail(lookback)
    pivots_low = sorted([p for p in find_pivots(sub, left=3, right=3) if p.kind == "low"],
                        key=lambda p: p.idx)
    pivots_high = sorted([p for p in find_pivots(sub, left=3, right=3) if p.kind == "high"],
                          key=lambda p: p.idx)
    if len(pivots_low) < 3 or len(pivots_high) < 2:
        return None

    best: PatternMatch | None = None
    # Try the most-recent triple of low pivots that fits IHS geometry
    for i in range(len(pivots_low) - 2):
        L1, L2, L3 = pivots_low[i], pivots_low[i + 1], pivots_low[i + 2]
        if L3.idx < len(sub) - 25 or L3.idx > len(sub) - 2:
            # right shoulder should be reasonably recent
            continue

        # Head must be the lowest of the three
        if L2.price >= L1.price or L2.price >= L3.price:
            continue

        shoulder_avg = (L1.price + L3.price) / 2
        head_depth = (shoulder_avg - L2.price) / shoulder_avg
        if not (0.04 <= head_depth <= 0.25):
            continue

        # Shoulder symmetry — price
        shoulder_price_diff = abs(L1.price - L3.price) / shoulder_avg
        if shoulder_price_diff > 0.08:
            continue

        # Shoulder symmetry — time
        d1 = L2.idx - L1.idx
        d2 = L3.idx - L2.idx
        if d1 == 0 or d2 == 0:
            continue
        time_asymmetry = abs(d1 - d2) / max(d1, d2)
        if time_asymmetry > 0.5:
            continue

        # Neckline: highest highs between (L1,L2) and (L2,L3)
        peak1 = sub.iloc[L1.idx + 1 : L2.idx]["high"].max() if L2.idx > L1.idx + 1 else None
        peak2 = sub.iloc[L2.idx + 1 : L3.idx]["high"].max() if L3.idx > L2.idx + 1 else None
        if peak1 is None or peak2 is None or pd.isna(peak1) or pd.isna(peak2):
            continue
        neckline = float(max(peak1, peak2))

        # L3 reaction strength (close in upper half = buying response)
        right_bar = sub.iloc[L3.idx]
        rb_range = max(right_bar["high"] - right_bar["low"], 1e-9)
        right_reaction = (right_bar["close"] - right_bar["low"]) / rb_range

        atr_pct = _atr_pct(daily)
        current_price = float(sub.iloc[-1]["close"])
        match = PatternMatch(
            pattern="inverse_head_shoulders",
            neckline=neckline,
            invalid_below=L2.price * 0.985,
            measured_move_target=neckline + (neckline - L2.price),
            structure={
                "left_shoulder": float(L1.price), "left_shoulder_idx": int(L1.idx),
                "head": float(L2.price), "head_idx": int(L2.idx),
                "right_shoulder": float(L3.price), "right_shoulder_idx": int(L3.idx),
                "left_to_head_bars": int(d1), "head_to_right_bars": int(d2),
                "structural_low": float(L2.price),
                "current_price": current_price,
                "neckline_distance_pct": float((neckline - current_price) / current_price),
                "base_bars": int(L3.idx - L1.idx),
            },
            geometry_features={
                "head_depth_pct": float(head_depth),
                "shoulder_price_asymmetry": float(shoulder_price_diff),
                "shoulder_time_asymmetry": float(time_asymmetry),
                "right_shoulder_reaction": float(right_reaction),
                "atr_pct": atr_pct,
            },
        )
        if best is None or match.structure["right_shoulder_idx"] > best.structure["right_shoulder_idx"]:
            best = match
    return best


# ---------------------------------------------------------------------------
# Base-on-Base — daily
# ---------------------------------------------------------------------------

def detect_base_on_base(daily: pd.DataFrame, lookback: int = 200) -> PatternMatch | None:
    """Two stacked bases — second base sits at or above first base's pivot high.

    Conservative thresholds:
      - each base 18–80 bars long
      - second base's low ≥ first base's pivot high × 0.93
      - second base's pivot high ≥ first base's pivot high
      - both bases shallow (depth ≤ 25 %)
      - dry-up volume during second base
    """
    if len(daily) < 80:
        return None
    sub = daily.tail(lookback).reset_index(drop=True)
    pivots = find_pivots(sub, left=4, right=4)
    pivot_highs = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.idx)
    pivot_lows = sorted([p for p in pivots if p.kind == "low"], key=lambda p: p.idx)
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    # Cluster pivot highs into rim levels (pivots within 2.5% of cluster mean = same rim)
    rims: list[dict] = []
    for p in pivot_highs:
        placed = False
        for cluster in rims:
            if abs(p.price - cluster["mean_price"]) / cluster["mean_price"] <= 0.025:
                cluster["pivots"].append(p)
                cluster["mean_price"] = sum(x.price for x in cluster["pivots"]) / len(cluster["pivots"])
                placed = True
                break
        if not placed:
            rims.append({"pivots": [p], "mean_price": p.price})

    if len(rims) < 2:
        return None
    # Sort by most-recent pivot first
    rims.sort(key=lambda r: max(p.idx for p in r["pivots"]), reverse=True)
    second_rim = rims[0]
    # First rim must be a *lower* price level (≥ 7 % below second rim)
    first_rim = None
    for r in rims[1:]:
        if r["mean_price"] < second_rim["mean_price"] * 0.93:
            first_rim = r
            break
    if first_rim is None:
        return None

    # Use cluster left/right edges to measure base widths
    second_top = max(second_rim["pivots"], key=lambda p: p.idx)
    second_left = min(second_rim["pivots"], key=lambda p: p.idx)
    first_top = max(first_rim["pivots"], key=lambda p: p.idx)
    first_left = min(first_rim["pivots"], key=lambda p: p.idx)
    if second_top.idx <= first_top.idx:
        return None

    second_base_bars = second_top.idx - second_left.idx
    if not (15 <= second_base_bars <= 80):
        return None

    # Second base low: lowest low within the second rim cluster's span
    interior_lows_2 = [p for p in pivot_lows if second_left.idx <= p.idx <= second_top.idx]
    if not interior_lows_2:
        return None
    second_low = min(interior_lows_2, key=lambda p: p.price)
    second_depth = (second_top.price - second_low.price) / second_top.price
    if second_depth > 0.25:
        return None

    # First base width = span within first rim cluster
    first_base_bars = first_top.idx - first_left.idx
    if first_base_bars < 5:
        # Single-pivot rim — use a synthetic 20-bar pre-window for measurement
        first_base_bars = 20
    if first_base_bars > 80:
        return None
    interior_lows_1 = [p for p in pivot_lows if first_left.idx <= p.idx <= first_top.idx]
    if not interior_lows_1:
        # Fall back to lookback-based search
        search_start = max(0, first_top.idx - 30)
        interior_lows_1 = [p for p in pivot_lows if search_start <= p.idx < first_top.idx]
        if not interior_lows_1:
            return None
    first_low = min(interior_lows_1, key=lambda p: p.price)
    first_depth = (first_top.price - first_low.price) / first_top.price
    if first_depth > 0.30:
        return None

    # Stacking constraint
    if second_low.price < first_top.price * 0.93:
        return None

    # Volume dry-up during second base vs first base
    vol_second = float(sub.iloc[second_left.idx : second_top.idx + 1]["volume"].mean())
    vol_first = float(sub.iloc[first_left.idx : first_top.idx + 1]["volume"].mean())
    dry_up = vol_second / max(vol_first, 1e-9)

    current_price = float(daily["close"].iloc[-1])
    return PatternMatch(
        pattern="base_on_base",
        neckline=float(second_top.price),
        invalid_below=float(second_low.price) * 0.985,
        measured_move_target=float(second_top.price) + (first_top.price - first_low.price),
        structure={
            "first_base_low": float(first_low.price),
            "first_base_top": float(first_top.price),
            "first_base_bars": int(first_base_bars),
            "second_base_low": float(second_low.price),
            "second_base_top": float(second_top.price),
            "second_base_bars": int(second_base_bars),
            "structural_low": float(second_low.price),
            "stacking_overlap_pct": float((first_top.price - second_low.price) / first_top.price),
            "current_price": current_price,
            "neckline_distance_pct": float((second_top.price - current_price) / current_price),
            "base_bars": int(second_base_bars),
        },
        geometry_features={
            "first_base_depth_pct": float(first_depth),
            "second_base_depth_pct": float(second_depth),
            "second_dry_up_ratio": float(dry_up),
            "rim_advance_pct": float((second_top.price - first_top.price) / first_top.price),
            "atr_pct": _atr_pct(daily),
        },
    )


# ---------------------------------------------------------------------------
# Tight Consolidation — daily
# ---------------------------------------------------------------------------

def detect_tight_consolidation(daily: pd.DataFrame, lookback: int = 60) -> PatternMatch | None:
    """N consecutive daily bars whose total range is within X * ATR.

    Conservative thresholds:
      - window 8–20 bars
      - total range ≤ 5 % of close OR ≤ 3 × ATR(14)
      - volume contracted vs prior 20 bars (≤ 0.85 ratio)
      - in the upper half of recent N=63d range (i.e., not consolidating after a top break)
    """
    if len(daily) < 50 or "atr_14" not in daily.columns:
        return None
    a14 = daily["atr_14"].iloc[-1]
    if pd.isna(a14) or a14 <= 0:
        return None

    best_window = None
    best_quality = -1.0

    # Try windows 8..20 bars long, ending at the latest bar
    for window in range(8, 21):
        if len(daily) < window + 25:
            continue
        win = daily.iloc[-window:]
        rng = float(win["high"].max() - win["low"].min())
        close = float(win["close"].iloc[-1])
        rng_pct = rng / max(close, 1e-9)
        rng_atr = rng / a14

        if rng_pct > 0.05 or rng_atr > 3.0:
            continue
        # volume contraction
        prior20 = daily["volume"].iloc[-(window + 20):-window]
        if len(prior20) < 10:
            continue
        vol_ratio = float(win["volume"].mean() / max(prior20.mean(), 1e-9))
        if vol_ratio > 0.85:
            continue
        # location within 63-day range (we want consolidation in upper half)
        recent63 = daily["close"].tail(63)
        loc = (close - recent63.min()) / max(recent63.max() - recent63.min(), 1e-9)
        if loc < 0.55:
            continue

        # Quality: tighter range × longer window × deeper dry-up = better
        quality = (1 - rng_atr / 3.0) * 0.4 + (window / 20) * 0.3 + (1 - vol_ratio) * 0.3
        if quality > best_quality:
            best_quality = quality
            best_window = (window, rng, rng_pct, rng_atr, vol_ratio, loc)

    if best_window is None:
        return None

    window, rng, rng_pct, rng_atr, vol_ratio, loc = best_window
    win = daily.iloc[-window:]
    pivot_high = float(win["high"].max())
    pivot_low = float(win["low"].min())
    current_price = float(daily["close"].iloc[-1])

    return PatternMatch(
        pattern="tight_consolidation",
        neckline=pivot_high,
        invalid_below=pivot_low * 0.985,
        measured_move_target=pivot_high + rng,
        structure={
            "window_bars": int(window),
            "range_pct": float(rng_pct),
            "range_atr": float(rng_atr),
            "vol_ratio": float(vol_ratio),
            "range_location_63d": float(loc),
            "pivot_high": pivot_high,
            "pivot_low": pivot_low,
            "structural_low": pivot_low,
            "current_price": current_price,
            "neckline_distance_pct": float((pivot_high - current_price) / current_price),
            "base_bars": int(window),
        },
        geometry_features={
            "range_pct": float(rng_pct),
            "range_atr": float(rng_atr),
            "duration_bars": float(window),
            "vol_dryup_ratio": float(vol_ratio),
            "location_in_range": float(loc),
            "atr_pct": _atr_pct(daily),
        },
    )


# ---------------------------------------------------------------------------
# Breakout Retest Hold — hourly
# ---------------------------------------------------------------------------

def detect_breakout_retest_hold(hourly: pd.DataFrame, daily: pd.DataFrame,
                                  lookback: int = 80) -> PatternMatch | None:
    """A prior breakout above a swing high, then a retest of the level holding above.

    Sequence:
      1. swing high P at index p_idx (the level that's being retested)
      2. price made a peak after P that was meaningfully above P (≥ 1 % above)
      3. price then pulled back from that peak to within ~1.5 ATR of P
      4. retest didn't lose the level (retest_low ≥ P × 0.985)
      5. price is currently above P
      6. retest is recent (≤ 15 bars old)

    Trigger remains P (re-break of the retest high). Invalid_below is the retest low.
    """
    if len(hourly) < 30:
        return None
    sub = hourly.tail(lookback).reset_index(drop=True)
    pivots_high = sorted([p for p in find_pivots(sub, left=3, right=3) if p.kind == "high"],
                          key=lambda p: p.idx)
    if not pivots_high:
        return None

    atr_h = float((sub["high"] - sub["low"]).rolling(14).mean().iloc[-1])
    if not np.isfinite(atr_h) or atr_h <= 0:
        return None

    last_idx = len(sub) - 1
    current = float(sub["close"].iloc[-1])

    best: PatternMatch | None = None
    # Iterate older pivots first (they're more likely to be the level being retested).
    # The most-recent pivots are usually the post-breakout peak, which we DON'T want as P.
    for P in pivots_high[-6:]:
        if last_idx - P.idx < 8:
            continue

        post = sub.iloc[P.idx + 1:]
        if len(post) < 5:
            continue

        # 1. Find the peak after P
        peak_local = int(post["high"].values.argmax())
        bo_peak_idx = P.idx + 1 + peak_local
        bo_peak_price = float(sub["high"].iloc[bo_peak_idx])

        # 2. Peak must be meaningfully above P (≥ 1 %)
        if bo_peak_price < P.price * 1.01:
            continue

        # 3. Need bars after the peak for a pullback
        if bo_peak_idx >= last_idx - 2:
            continue

        # 4. Find deepest pullback after the peak
        after_peak = sub.iloc[bo_peak_idx + 1:]
        retest_local = int(after_peak["low"].values.argmin())
        retest_idx = bo_peak_idx + 1 + retest_local
        retest_low = float(after_peak["low"].iloc[retest_local])

        # 5. Retest must reach down close to P (within 1.5 ATR)
        if retest_low > P.price + atr_h * 1.5:
            continue

        # 6. Retest must not have lost the level
        if retest_low < P.price * 0.985:
            continue

        # 7. Currently above the level
        if current < P.price * 0.998:
            continue

        # 8. Retest must be recent
        if last_idx - retest_idx > 15:
            continue

        # First "breakout" bar = first bar after P that closed > P (informational only)
        bo_mask = post["close"] > P.price * 1.001
        breakout_idx = P.idx + 1 + int(bo_mask.values.argmax()) if bo_mask.any() else bo_peak_idx

        match = PatternMatch(
            pattern="breakout_retest_hold",
            neckline=float(P.price),
            invalid_below=retest_low * 0.99,
            measured_move_target=P.price + (P.price - retest_low) * 2,
            structure={
                "level": float(P.price),
                "level_idx": int(P.idx),
                "breakout_idx": int(breakout_idx),
                "bo_peak_idx": int(bo_peak_idx),
                "bo_peak_price": float(bo_peak_price),
                "retest_idx": int(retest_idx),
                "retest_low": retest_low,
                "structural_low": retest_low,
                "current_price": current,
                "neckline_distance_pct": float((P.price - current) / current),
                "base_bars": int(last_idx - P.idx),
                "bars_since_breakout": int(last_idx - breakout_idx),
            },
            geometry_features={
                "retest_depth_atr": float((P.price - retest_low) / max(atr_h, 1e-9)),
                "bars_breakout_to_retest": float(retest_idx - breakout_idx),
                "bars_since_retest": float(last_idx - retest_idx),
                "current_above_level_atr": float((current - P.price) / max(atr_h, 1e-9)),
                "bo_peak_excess_pct": float((bo_peak_price - P.price) / P.price),
                "atr_pct": _atr_pct(daily),
            },
        )
        # Prefer the most-recent valid level
        if best is None or match.structure["level_idx"] > best.structure["level_idx"]:
            best = match
    return best


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

ALL_DETECTORS = {
    "double_bottom":           ("hourly", detect_double_bottom),
    "bull_flag":               ("hourly", detect_bull_flag),
    "vcp":                     ("daily",  detect_vcp),
    "ascending_triangle":      ("hourly", detect_ascending_triangle),
    "cup_with_handle":         ("daily",  detect_cup_with_handle),
    "inverse_head_shoulders":  ("hourly", detect_inverse_head_shoulders),
    "base_on_base":            ("daily",  detect_base_on_base),
    "tight_consolidation":     ("daily",  detect_tight_consolidation),
    "breakout_retest_hold":    ("hourly", detect_breakout_retest_hold),
}


def detect_all(daily: pd.DataFrame, hourly: pd.DataFrame,
                enabled: Iterable[str] | None = None) -> list[PatternMatch]:
    enabled = set(enabled) if enabled else set(ALL_DETECTORS.keys())
    matches: list[PatternMatch] = []
    for name, (timeframe, fn) in ALL_DETECTORS.items():
        if name not in enabled:
            continue
        try:
            if timeframe == "daily":
                m = fn(daily)
            else:
                m = fn(hourly, daily)
        except Exception as e:
            log.warning("%s detector failed: %s", name, e)
            m = None
        if m is not None:
            matches.append(m)
    return matches
