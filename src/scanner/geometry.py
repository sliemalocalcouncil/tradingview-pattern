"""Pattern-specific geometry scores (0–100) — per design §5.P.

Each scorer takes a PatternMatch (already detected) and returns a 0–100
quality score plus a feature breakdown for explainability.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .indicators import bell_score, clamp, lerp_score
from .models import PatternMatch


def _normalize(score: float) -> float:
    return float(round(clamp(score, 0, 100), 2))


# ---------------------------------------------------------------------------
# Double Bottom geometry — §5.P.1
# ---------------------------------------------------------------------------

def double_bottom_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    s = match.structure
    out: dict[str, Any] = {}

    # 1. low similarity
    sim_pct = f.get("low_similarity_pct", 0.05)
    sub_low_sim = lerp_score(sim_pct, 0.05, 0.005)  # 5% → 0, 0.5% → 100 (invert via swapped bounds)
    sub_low_sim = clamp((1 - sim_pct / 0.05) * 100, 0, 100)
    out["sub_low_similarity"] = round(sub_low_sim, 1)

    # 2. bounce quality
    bounce = f.get("bounce_pct", 0.03)
    sub_bounce = clamp((bounce - 0.03) / 0.07 * 100, 0, 100)  # 3% → 0, 10%+ → 100
    out["sub_bounce_quality"] = round(sub_bounce, 1)

    # 3. separation fit (10–30 bars sweet spot)
    sep = s.get("separation_bars", 0)
    sub_sep = bell_score(sep, 10, 30, falloff=15)
    out["sub_separation_fit"] = round(sub_sep, 1)

    # 4. right-bottom reaction (close in upper half of bar = buying response)
    rbr = f.get("right_bottom_reaction", 0.5)
    sub_rbr = clamp(rbr * 100, 0, 100)
    out["sub_right_bottom_reaction"] = round(sub_rbr, 1)

    # 5. base context: formed above the daily 50-EMA?
    sub_ctx = 50.0
    if "ema_50" in daily.columns and pd.notna(daily["ema_50"].iloc[-1]):
        if daily["close"].iloc[-1] > daily["ema_50"].iloc[-1]:
            sub_ctx = 100.0
        else:
            sub_ctx = 30.0
    out["sub_base_context"] = sub_ctx

    weights = {"sub_low_similarity": 0.30, "sub_bounce_quality": 0.20,
               "sub_separation_fit": 0.15, "sub_right_bottom_reaction": 0.15,
               "sub_base_context": 0.20}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# Bull Flag / HTP geometry — §5.P.3 / §5.P.4
# ---------------------------------------------------------------------------

def bull_flag_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    s = match.structure
    out: dict[str, Any] = {}

    pole_atr_mult = f.get("pole_atr_multiple", 0.0)
    sub_pole = lerp_score(pole_atr_mult, 1.0, 5.0)
    out["sub_pole_impulse"] = round(sub_pole, 1)

    flag_range = f.get("flag_range_pct", 0.20)
    pole_gain = max(f.get("pole_gain", 0.10), 1e-9)
    flag_tightness_ratio = flag_range / pole_gain
    sub_tight = clamp((1 - flag_tightness_ratio / 0.6) * 100, 0, 100)
    out["sub_flag_tightness"] = round(sub_tight, 1)

    pullback = f.get("pullback_pct", 0.10)
    pullback_ratio = pullback / pole_gain
    sub_shallow = clamp((1 - pullback_ratio / 0.5) * 100, 0, 100)
    out["sub_flag_shallowness"] = round(sub_shallow, 1)

    drift = f.get("flag_drift_per_bar", -0.005)
    # ideal drift around 0 (slight up or flat), down-drift = penalty
    sub_drift = clamp(100 - max(0, -drift) * 5000, 0, 100)
    out["sub_flag_drift"] = round(sub_drift, 1)

    flag_bars = s.get("flag_bars", 0)
    if match.pattern == "high_tight_pullback":
        sub_dur = bell_score(flag_bars, 3, 10, falloff=5)
    else:
        sub_dur = bell_score(flag_bars, 5, 20, falloff=8)
    out["sub_flag_duration"] = round(sub_dur, 1)

    weights = {"sub_pole_impulse": 0.30, "sub_flag_tightness": 0.25,
               "sub_flag_shallowness": 0.20, "sub_flag_drift": 0.15,
               "sub_flag_duration": 0.10}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# VCP geometry — §5.P.5
# ---------------------------------------------------------------------------

def vcp_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    s = match.structure
    out: dict[str, Any] = {}

    # Sequence length sweet spot 3–6
    n = int(s.get("n_contractions", 0))
    sub_n = bell_score(n, 3, 6, falloff=3)
    out["sub_contraction_count"] = round(sub_n, 1)

    # Reduction quality: each ≤ 0.9 × prior
    contractions = s.get("contractions_pct", [])
    if len(contractions) >= 2:
        ratios = [contractions[i + 1] / max(contractions[i], 1e-9) for i in range(len(contractions) - 1)]
        avg_ratio = float(sum(ratios) / len(ratios))
        sub_red = clamp((1 - avg_ratio) / 0.5 * 100, 0, 100)
    else:
        avg_ratio = 1.0
        sub_red = 0.0
    out["sub_reduction_rate"] = round(sub_red, 1)

    # Final pivot tightness: last contraction ≤ 8% → 100
    last_pct = float(s.get("contractions_pct", [10])[-1]) if s.get("contractions_pct") else 10
    sub_pivot_tight = clamp((10 - last_pct) / 7 * 100, 0, 100)
    out["sub_pivot_tightness"] = round(sub_pivot_tight, 1)

    # Volume dry-up (already in V; here we use it as a geometry hint too)
    dry = float(s.get("dry_up_ratio", 1.0))
    sub_vol = clamp((1.0 - dry) / 0.5 * 100, 0, 100)  # 50% dry-up → 100
    out["sub_volume_dryup"] = round(sub_vol, 1)

    # Base depth fit: sweet spot 15–30 %
    depth = float(f.get("base_depth_pct", 0.20)) * 100
    sub_depth = bell_score(depth, 15, 30, falloff=10)
    out["sub_base_depth"] = round(sub_depth, 1)

    # Base duration in bars
    base_bars = int(s.get("base_bars", 0))
    sub_dur = bell_score(base_bars, 25, 125, falloff=40)  # 5–25 weeks ≈ 25–125 bars
    out["sub_base_duration"] = round(sub_dur, 1)

    weights = {"sub_contraction_count": 0.25, "sub_reduction_rate": 0.25,
               "sub_pivot_tightness": 0.15, "sub_volume_dryup": 0.15,
               "sub_base_depth": 0.10, "sub_base_duration": 0.10}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# Ascending Triangle geometry — §5.P.2
# ---------------------------------------------------------------------------

def ascending_triangle_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    s = match.structure
    out: dict[str, Any] = {}

    touches = int(f.get("top_touches", 2))
    sub_top = clamp((touches - 1) / 3 * 100, 0, 100)  # 1 → 0, 4 → 100
    out["sub_flat_top_quality"] = round(sub_top, 1)

    slope = float(f.get("low_slope", 0))
    last_low = float(s.get("lowest_low", 1))
    slope_pct_per_bar = slope / max(last_low, 1e-9) * 100
    # 0 → 0, 0.3% per bar → 100
    sub_slope = clamp(slope_pct_per_bar / 0.3 * 100, 0, 100)
    out["sub_hl_slope"] = round(sub_slope, 1)

    compression = float(f.get("compression", 0))
    sub_comp = clamp(compression / 0.5 * 100, 0, 100)
    out["sub_compression"] = round(sub_comp, 1)

    base_bars = int(s.get("base_bars", 0))
    sub_dur = bell_score(base_bars, 20, 60, falloff=20)
    out["sub_duration"] = round(sub_dur, 1)

    weights = {"sub_flat_top_quality": 0.30, "sub_hl_slope": 0.30,
               "sub_compression": 0.25, "sub_duration": 0.15}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# Cup with Handle geometry — §5.P.6
# ---------------------------------------------------------------------------

def cup_with_handle_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    s = match.structure
    out: dict[str, Any] = {}

    # cup depth — sweet spot 18–28 %
    cd = f.get("cup_depth_pct", 0.20) * 100
    out["sub_cup_depth"] = round(bell_score(cd, 18, 28, falloff=10), 1)

    # cup duration — sweet spot 35–80 daily bars
    dur = float(f.get("cup_duration_bars", 50))
    out["sub_cup_duration"] = round(bell_score(dur, 35, 80, falloff=25), 1)

    # rim symmetry — closer = better
    sym = f.get("rim_symmetry_pct", 0.05)
    out["sub_rim_symmetry"] = round(clamp((0.05 - sym) / 0.05 * 100, 0, 100), 1)

    # handle depth — shallow is better
    hd = f.get("handle_depth_pct", 0.10) * 100
    out["sub_handle_shallowness"] = round(clamp((12 - hd) / 12 * 100, 0, 100), 1)

    # handle/cup ratio — sweet spot 8–22 %
    hcr = float(f.get("handle_to_cup_ratio", 0.15)) * 100
    out["sub_handle_proportion"] = round(bell_score(hcr, 8, 22, falloff=12), 1)

    # U-shape score (>= 0.4 means the bottom is well separated from rims)
    us = float(f.get("u_shape_score", 0.5))
    out["sub_u_shape"] = round(clamp(us / 0.5 * 100, 0, 100), 1)

    weights = {"sub_cup_depth": 0.20, "sub_cup_duration": 0.15, "sub_rim_symmetry": 0.20,
               "sub_handle_shallowness": 0.20, "sub_handle_proportion": 0.10, "sub_u_shape": 0.15}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# Inverse H&S geometry — §5.P.7
# ---------------------------------------------------------------------------

def inverse_head_shoulders_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    out: dict[str, Any] = {}

    # head depth — sweet spot 8–20 %
    hd = f.get("head_depth_pct", 0.10) * 100
    out["sub_head_depth"] = round(bell_score(hd, 8, 20, falloff=10), 1)

    # shoulder price symmetry — closer = better
    sps = f.get("shoulder_price_asymmetry", 0.05)
    out["sub_shoulder_price_sym"] = round(clamp((0.08 - sps) / 0.08 * 100, 0, 100), 1)

    # shoulder time symmetry
    sts = f.get("shoulder_time_asymmetry", 0.3)
    out["sub_shoulder_time_sym"] = round(clamp((0.5 - sts) / 0.5 * 100, 0, 100), 1)

    # right shoulder reaction (close in upper half = buying)
    rsr = float(f.get("right_shoulder_reaction", 0.5))
    out["sub_right_shoulder_reaction"] = round(clamp(rsr * 100, 0, 100), 1)

    # base context
    sub_ctx = 50.0
    if "ema_50" in daily.columns and pd.notna(daily["ema_50"].iloc[-1]):
        if daily["close"].iloc[-1] > daily["ema_50"].iloc[-1]:
            sub_ctx = 100.0
        else:
            sub_ctx = 30.0
    out["sub_base_context"] = sub_ctx

    weights = {"sub_head_depth": 0.20, "sub_shoulder_price_sym": 0.30,
               "sub_shoulder_time_sym": 0.15, "sub_right_shoulder_reaction": 0.20,
               "sub_base_context": 0.15}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# Base-on-Base geometry — §5.P.8
# ---------------------------------------------------------------------------

def base_on_base_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    s = match.structure
    out: dict[str, Any] = {}

    # second base shallower than first = strength
    fbd = f.get("first_base_depth_pct", 0.20)
    sbd = f.get("second_base_depth_pct", 0.20)
    if fbd > 0:
        ratio = sbd / fbd
        out["sub_shallower_second"] = round(clamp((1.0 - ratio) / 0.5 * 100, 0, 100), 1)
    else:
        out["sub_shallower_second"] = 50.0

    # second base depth — sweet spot 8–18 %
    sbd_pct = sbd * 100
    out["sub_second_base_depth"] = round(bell_score(sbd_pct, 8, 18, falloff=10), 1)

    # rim advance — second top higher than first = continuation strength
    rim_adv = f.get("rim_advance_pct", 0.0)
    out["sub_rim_advance"] = round(clamp(rim_adv / 0.10 * 100, 0, 100), 1)

    # dry-up during second base
    dryup = f.get("second_dry_up_ratio", 1.0)
    out["sub_dry_up"] = round(clamp((1.0 - dryup) / 0.5 * 100, 0, 100), 1)

    # second base duration — sweet spot 25–60 bars
    bb = float(s.get("second_base_bars", 30))
    out["sub_second_base_duration"] = round(bell_score(bb, 25, 60, falloff=25), 1)

    weights = {"sub_shallower_second": 0.25, "sub_second_base_depth": 0.20,
               "sub_rim_advance": 0.20, "sub_dry_up": 0.20,
               "sub_second_base_duration": 0.15}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# Tight Consolidation geometry — §5.P.9
# ---------------------------------------------------------------------------

def tight_consolidation_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    out: dict[str, Any] = {}

    # tightness in ATR — under 1.5 ATR = excellent
    rng_atr = f.get("range_atr", 3.0)
    out["sub_tightness"] = round(clamp((3.0 - rng_atr) / 3.0 * 100, 0, 100), 1)

    # duration — sweet spot 12–18 bars
    dur = float(f.get("duration_bars", 10))
    out["sub_duration"] = round(bell_score(dur, 12, 18, falloff=8), 1)

    # volume dry-up — < 0.7 = strong
    vd = f.get("vol_dryup_ratio", 1.0)
    out["sub_dry_up"] = round(clamp((0.85 - vd) / 0.5 * 100, 0, 100), 1)

    # location in 63-day range — closer to top = better setup
    loc = f.get("location_in_range", 0.5)
    out["sub_location"] = round(clamp((loc - 0.55) / 0.4 * 100, 0, 100), 1)

    weights = {"sub_tightness": 0.40, "sub_duration": 0.15, "sub_dry_up": 0.25,
               "sub_location": 0.20}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# Breakout Retest Hold geometry — §5.P.10
# ---------------------------------------------------------------------------

def breakout_retest_hold_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    f = match.geometry_features
    out: dict[str, Any] = {}

    # retest depth in ATR — sweet spot 0.3–1.0 ATR
    rd = f.get("retest_depth_atr", 0.5)
    out["sub_retest_depth"] = round(bell_score(rd, 0.3, 1.0, falloff=0.7), 1)

    # bars breakout → retest — sweet spot 2–8 bars
    bbr = float(f.get("bars_breakout_to_retest", 4))
    out["sub_retest_timing"] = round(bell_score(bbr, 2, 8, falloff=5), 1)

    # bars since retest — fresher is better, but not too fresh
    bsr = float(f.get("bars_since_retest", 3))
    out["sub_retest_freshness"] = round(bell_score(bsr, 1, 6, falloff=4), 1)

    # current location vs the level — moderate above is best (0.2–1.0 ATR)
    cal = f.get("current_above_level_atr", 0.5)
    out["sub_hold_quality"] = round(bell_score(cal, 0.2, 1.0, falloff=0.8), 1)

    weights = {"sub_retest_depth": 0.30, "sub_retest_timing": 0.20,
               "sub_retest_freshness": 0.20, "sub_hold_quality": 0.30}
    total = sum(out[k] * w for k, w in weights.items())
    out["_score"] = _normalize(total)
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GEOMETRY_SCORERS = {
    "double_bottom":          double_bottom_geometry,
    "bull_flag":              bull_flag_geometry,
    "high_tight_pullback":    bull_flag_geometry,
    "vcp":                    vcp_geometry,
    "ascending_triangle":     ascending_triangle_geometry,
    "cup_with_handle":        cup_with_handle_geometry,
    "inverse_head_shoulders": inverse_head_shoulders_geometry,
    "base_on_base":           base_on_base_geometry,
    "tight_consolidation":    tight_consolidation_geometry,
    "breakout_retest_hold":   breakout_retest_hold_geometry,
}


def score_geometry(match: PatternMatch, daily: pd.DataFrame) -> dict[str, Any]:
    fn = GEOMETRY_SCORERS.get(match.pattern)
    if fn is None:
        return {"_score": 50.0, "note": "no_geometry_scorer"}
    return fn(match, daily)
