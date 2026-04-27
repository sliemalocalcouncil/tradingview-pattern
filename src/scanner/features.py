"""Feature scoring — 9 common dimensions, each broken into sub-features.

Implements the explainable scoring per design doc §5.3 / §6.
Each function returns a dict with both the raw measurement AND the 0–100 score,
so the full breakdown can be persisted to features_json for post-hoc analysis.

Design principle: NO max() collapsing across independent signals. Each
sub-feature is computed and weighted explicitly (fixes finding C6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .indicators import Level, bell_score, clamp, find_levels, lerp_score, nearest_levels


# ---------------------------------------------------------------------------
# T. Trend Regime (§6.T)
# ---------------------------------------------------------------------------

def trend_regime(daily: pd.DataFrame, market_daily: pd.DataFrame | None = None) -> dict[str, Any]:
    """Continuous 0–100 trend score; 6 sub-features, weighted.

    market_daily — SPY (or QQQ) daily bars to compute relative strength against.
    """
    out: dict[str, Any] = {}
    last = daily.iloc[-1]

    # T1: EMA stack
    flags = [
        bool(last["close"] > last["ema_20"]),
        bool(last["ema_20"] > last["ema_50"]),
        bool(last["ema_50"] > last["ema_200"]),
    ]
    t1 = sum(flags) / 3 * 100
    out["T1_ema_stack"] = {"value": flags, "score": t1}

    # T2: EMA50 slope (% per day)
    if len(daily) >= 22:
        e = daily["ema_50"].tail(21).values
        slope = float(np.polyfit(range(21), e, 1)[0])
        slope_pct = slope / max(last["close"], 1e-9) * 100
        # Map 0%→50, 0.3% (per day) → 100, -0.3% → 0
        t2 = clamp(50 + slope_pct / 0.3 * 50, 0, 100)
    else:
        slope_pct = 0.0
        t2 = 50.0
    out["T2_ema50_slope"] = {"value": slope_pct, "score": t2}

    # T3: 63-day high proximity
    if len(daily) >= 63:
        ratio = float(last["close"] / daily["close"].tail(63).max())
        t3 = clamp((ratio - 0.80) / 0.20 * 100, 0, 100)
    else:
        ratio = 0.0
        t3 = 50.0
    out["T3_high_proximity"] = {"value": ratio, "score": t3}

    # T4 / T5: relative strength vs market index (simple absolute comparison; v2 will percentile-rank)
    t4 = t5 = 50.0
    rs_1m = rs_3m = None
    if market_daily is not None and len(daily) >= 64 and len(market_daily) >= 64:
        try:
            r1 = daily["close"].iloc[-1] / daily["close"].iloc[-22] - 1
            m1 = market_daily["close"].iloc[-1] / market_daily["close"].iloc[-22] - 1
            rs_1m = float(r1 - m1)
            # outperform by +0% → 50, +5% → 100, -5% → 0
            t4 = clamp(50 + rs_1m * 1000, 0, 100)
            r3 = daily["close"].iloc[-1] / daily["close"].iloc[-64] - 1
            m3 = market_daily["close"].iloc[-1] / market_daily["close"].iloc[-64] - 1
            rs_3m = float(r3 - m3)
            t5 = clamp(50 + rs_3m * 500, 0, 100)
        except Exception:
            pass
    out["T4_rs_1m"] = {"value": rs_1m, "score": t4}
    out["T5_rs_3m"] = {"value": rs_3m, "score": t5}

    # T6: ADR healthy band [1.5%, 8%]
    if len(daily) >= 20:
        adr = float(((daily["high"] - daily["low"]) / daily["close"]).tail(20).mean() * 100)
        t6 = bell_score(adr, 1.5, 8.0, falloff=4.0)
    else:
        adr = 0.0
        t6 = 50.0
    out["T6_adr_healthy"] = {"value": adr, "score": t6}

    # Aggregate (weights from design §5.3)
    weights = {"T1_ema_stack": 0.25, "T2_ema50_slope": 0.20, "T3_high_proximity": 0.15,
               "T4_rs_1m": 0.20, "T5_rs_3m": 0.15, "T6_adr_healthy": 0.05}
    score = sum(out[k]["score"] * w for k, w in weights.items())
    out["_score"] = float(round(score, 2))
    return out


# ---------------------------------------------------------------------------
# VC. Volatility Compression (§6.VC)
# ---------------------------------------------------------------------------

def volatility_compression(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(daily) < 60:
        out["_score"] = 50.0
        return out

    # VC1: ATR(10) / ATR(50)
    atr10 = daily["close"].rolling(10).apply(lambda x: x.diff().abs().mean(), raw=False).iloc[-1]
    atr50 = daily["atr_50"].iloc[-1]
    if pd.notna(atr10) and pd.notna(atr50) and atr50 > 0:
        ratio = float(atr10 / atr50)
        vc1 = clamp((1.0 - ratio) / 0.4 * 100, 0, 100)
    else:
        ratio, vc1 = 1.0, 50.0
    out["VC1_atr_ratio"] = {"value": ratio, "score": vc1}

    # VC2: range_recent_10 vs range_prior_20
    rp = float(daily["range_pct"].tail(10).mean())
    rp20 = float(daily["range_pct"].iloc[-30:-10].mean()) if len(daily) >= 30 else rp
    rp_ratio = rp / max(rp20, 1e-9)
    vc2 = clamp((1.0 - rp_ratio) / 0.6 * 100, 0, 100)
    out["VC2_range_contraction"] = {"value": rp_ratio, "score": vc2}

    # VC3: 120-day range percentile (simple proxy for BB width without BB)
    if len(daily) >= 120:
        rng_today = float(daily["range_pct"].iloc[-1])
        pct = float((daily["range_pct"].tail(120) <= rng_today).mean())
        vc3 = (1 - pct) * 100  # smaller percentile = greater compression
    else:
        vc3 = 50.0
    out["VC3_range_pct_rank"] = {"score": vc3}

    weights = {"VC1_atr_ratio": 0.40, "VC2_range_contraction": 0.35, "VC3_range_pct_rank": 0.25}
    out["_score"] = float(round(sum(out[k]["score"] * w for k, w in weights.items()), 2))
    return out


# ---------------------------------------------------------------------------
# SR. Support/Resistance Quality (§6.SR)
# ---------------------------------------------------------------------------

def sr_quality(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    levels = find_levels(daily, lookback=120, atr_bw_mult=0.5, min_touches=2)
    if not levels:
        out["_score"] = 30.0
        out["_levels_found"] = 0
        return out

    price = float(daily["close"].iloc[-1])
    atr14 = float(daily["atr_14"].iloc[-1]) if pd.notna(daily["atr_14"].iloc[-1]) else 0.0
    sup, res = nearest_levels(price, levels)
    target = res or sup
    if target is None:
        out["_score"] = 40.0
        out["_levels_found"] = len(levels)
        return out

    # SR1 touches
    sr1 = clamp((target.touches - 1) / 4 * 100, 0, 100)
    out["SR1_touches"] = {"value": target.touches, "score": sr1}

    # SR2 cluster tightness (std/ATR)
    if atr14 > 0:
        tight = target.std / atr14
        sr2 = clamp((1.0 - tight / 0.5) * 100, 0, 100)
    else:
        sr2 = 50.0
    out["SR2_tightness"] = {"value": target.std, "score": sr2}

    # SR3 recency (bars since last touch)
    bars_since = len(daily) - 1 - target.last_touch_idx
    if bars_since <= 30:
        sr3 = 100.0
    elif bars_since >= 90:
        sr3 = 0.0
    else:
        sr3 = (90 - bars_since) / 60 * 100
    out["SR3_recency"] = {"value": int(bars_since), "score": sr3}

    # SR4 price location: closer to resistance = higher score for breakout setups
    if sup and res:
        loc = (price - sup.mid) / max(res.mid - sup.mid, 1e-9)
        sr4 = clamp(loc * 100, 0, 100)
    elif res:
        atr_dist = (res.mid - price) / max(atr14, 1e-9)
        sr4 = clamp(100 - atr_dist * 30, 0, 100)
    else:
        sr4 = 30.0
    out["SR4_price_location"] = {"score": sr4}

    # SR5 reaction strength (avg bounce in ATR)
    sr5 = clamp(target.reaction_avg_atr / 2.0 * 100, 0, 100)
    out["SR5_reaction"] = {"value": target.reaction_avg_atr, "score": sr5}

    weights = {"SR1_touches": 0.30, "SR2_tightness": 0.15, "SR3_recency": 0.15,
               "SR4_price_location": 0.25, "SR5_reaction": 0.15}
    out["_score"] = float(round(sum(out[k]["score"] * w for k, w in weights.items()), 2))
    out["_levels_found"] = len(levels)
    out["_target_level"] = round(target.mid, 4)
    return out


# ---------------------------------------------------------------------------
# V. Volume Confirmation (§6.V)
# ---------------------------------------------------------------------------

def volume_confirmation(daily: pd.DataFrame, hourly: pd.DataFrame,
                         pattern_match=None) -> dict[str, Any]:
    """Five sub-features. NO max() collapse — each computed independently."""
    out: dict[str, Any] = {}

    # V1: today's day RVOL vs 20d avg
    if len(daily) >= 20 and pd.notna(daily["rvol_20"].iloc[-1]):
        v1raw = float(daily["rvol_20"].iloc[-1])
        # 1.0 → 50, 1.5 → 75, 2.0 → 100; below 0.5 → 0
        v1 = clamp((v1raw - 0.5) / 1.5 * 100, 0, 100)
    else:
        v1raw, v1 = 1.0, 50.0
    out["V1_rvol_day"] = {"value": v1raw, "score": v1}

    # V2: same-slot RVOL (compare current hour to same hour-of-day across last N sessions)
    v2raw, v2, v2_status = _same_slot_rvol(hourly, min_samples=8)
    out["V2_rvol_sameslot"] = {"value": v2raw, "score": v2, "status": v2_status}

    # V3: most recent bar volume vs prior 10 bars
    if len(hourly) >= 12:
        recent = float(hourly["volume"].iloc[-1])
        prior = float(hourly["volume"].iloc[-11:-1].mean())
        v3raw = recent / max(prior, 1e-9)
        v3 = clamp((v3raw - 0.5) / 1.5 * 100, 0, 100)
    else:
        v3raw, v3 = 1.0, 50.0
    out["V3_breakout_bar"] = {"value": v3raw, "score": v3}

    # V4: base dry-up — pattern-aware if we know the base window
    v4raw, v4 = _base_dryup(daily, pattern_match)
    out["V4_base_dryup"] = {"value": v4raw, "score": v4}

    # V5: today's volume vs 20-day average (already in V1, but emphasizes expansion specifically)
    if len(daily) >= 21:
        cur = float(daily["volume"].iloc[-1])
        avg20 = float(daily["volume"].tail(20).mean())
        v5raw = cur / max(avg20, 1e-9)
        v5 = clamp((v5raw - 1.0) / 1.5 * 100, 0, 100)
    else:
        v5raw, v5 = 1.0, 50.0
    out["V5_bo_expansion"] = {"value": v5raw, "score": v5}

    weights = {"V1_rvol_day": 0.20, "V2_rvol_sameslot": 0.20, "V3_breakout_bar": 0.25,
               "V4_base_dryup": 0.20, "V5_bo_expansion": 0.15}
    out["_score"] = float(round(sum(out[k]["score"] * w for k, w in weights.items()), 2))
    return out


def _same_slot_rvol(hourly: pd.DataFrame, min_samples: int = 8) -> tuple[float | None, float, str]:
    if hourly.empty:
        return None, 50.0, "no_data"
    # Group by hour-of-day in the bar's local (UTC) timestamp; assume hourly index
    last_ts = hourly.index[-1]
    slot = last_ts.hour
    same_slot = hourly[hourly.index.hour == slot]
    if len(same_slot) <= 1:
        return None, 50.0, "no_history"
    history = same_slot.iloc[:-1].tail(20)
    if len(history) < min_samples:
        return None, 50.0, "insufficient_sample"
    avg_vol = float(history["volume"].mean())
    if avg_vol <= 0:
        return None, 50.0, "zero_avg"
    rvol_val = float(same_slot["volume"].iloc[-1] / avg_vol)
    score = clamp((rvol_val - 0.5) / 1.5 * 100, 0, 100)
    return rvol_val, score, "ok"


def _base_dryup(daily: pd.DataFrame, match) -> tuple[float, float]:
    """Compare base-window volume vs the equivalent window before the base."""
    base_bars = 20
    if match is not None:
        bb = match.structure.get("base_bars")
        if isinstance(bb, int) and 5 <= bb <= 60:
            base_bars = bb

    if len(daily) < base_bars * 2:
        return 1.0, 50.0
    base_vol = float(daily["volume"].tail(base_bars).mean())
    prior_vol = float(daily["volume"].iloc[-base_bars * 2 : -base_bars].mean())
    if prior_vol <= 0:
        return 1.0, 50.0
    ratio = base_vol / prior_vol
    # 0.5 (deep dry-up) → 100, 1.0 (no change) → 50, 1.5+ (expansion) → 0
    score = clamp(100 - (ratio - 0.5) * 100, 0, 100)
    return ratio, score


# ---------------------------------------------------------------------------
# BR. Breakout Readiness (§6.BR)
# ---------------------------------------------------------------------------

def breakout_readiness(daily: pd.DataFrame, hourly: pd.DataFrame, match) -> dict[str, Any]:
    out: dict[str, Any] = {}
    price = float(daily["close"].iloc[-1])
    atr14 = float(daily["atr_14"].iloc[-1]) if pd.notna(daily["atr_14"].iloc[-1]) else 0.0
    trigger = match.neckline if match else None

    # BR1: ATR-distance to trigger
    if trigger and atr14 > 0:
        d_atr = (trigger - price) / atr14
        out["_distance_atr"] = round(d_atr, 3)
        if d_atr > 0:
            br1 = clamp(100 - d_atr / 1.5 * 100, 0, 100)
        else:  # already broke out
            br1 = 100.0 if d_atr > -0.5 else clamp(60 + d_atr * 20, 0, 100)
    else:
        br1 = 50.0
    out["BR1_atr_distance"] = {"score": br1}

    # BR2: pre-breakout compression (last 5 vs prior 20 hourly range)
    if len(hourly) >= 25:
        rh = (hourly["high"] - hourly["low"]) / hourly["close"]
        recent5 = float(rh.tail(5).mean())
        prior20 = float(rh.iloc[-25:-5].mean())
        if prior20 > 0:
            ratio = recent5 / prior20
            br2 = clamp((1.0 - ratio) / 0.6 * 100, 0, 100)
        else:
            br2 = 50.0
    else:
        br2 = 50.0
    out["BR2_compression"] = {"score": br2}

    # BR3: retest hold quality (only meaningful for retest patterns)
    br3 = 50.0
    if match and "retest_low" in match.structure and trigger and atr14 > 0:
        retest_low = match.structure["retest_low"]
        gap_atr = (trigger - retest_low) / atr14
        # ≤ 0.5 ATR = perfect, ≥ 1.5 ATR = weak retest
        br3 = clamp(100 - gap_atr * 50, 0, 100)
    out["BR3_retest"] = {"score": br3}

    # BR4: bars since breakout (only if we've already broken out)
    br4 = 50.0
    if match and match.structure.get("bars_since_breakout") is not None:
        bsb = int(match.structure["bars_since_breakout"])
        if bsb <= 3:
            br4 = 100.0
        elif bsb <= 10:
            br4 = 100 - (bsb - 3) * 10
        else:
            br4 = 30.0
    out["BR4_bo_age"] = {"score": br4}

    weights = {"BR1_atr_distance": 0.40, "BR2_compression": 0.25, "BR3_retest": 0.20, "BR4_bo_age": 0.15}
    out["_score"] = float(round(sum(out[k]["score"] * w for k, w in weights.items()), 2))
    return out


# ---------------------------------------------------------------------------
# R. Risk Definition (§6.R)
# ---------------------------------------------------------------------------

def risk_definition(daily: pd.DataFrame, match) -> dict[str, Any]:
    out: dict[str, Any] = {}
    price = float(daily["close"].iloc[-1])
    atr14 = float(daily["atr_14"].iloc[-1]) if pd.notna(daily["atr_14"].iloc[-1]) else 0.0
    invalid = match.invalid_below if match else None
    target = match.measured_move_target if match else None

    if invalid is None:
        out["_score"] = 0.0
        out["_status"] = "no_invalid_below"
        return out

    # R1: stop clarity (proximity of invalid_below to a structural low)
    sl = match.structure.get("structural_low") if match else None
    if sl is not None and atr14 > 0:
        gap = abs(invalid - sl) / atr14
        r1 = clamp(100 - gap * 200, 0, 100)
    else:
        r1 = 70.0  # not great but not zero
    out["R1_stop_clarity"] = {"score": r1}

    # R2: stop distance % — sweet spot 2–8 %
    stop_pct = (price - invalid) / price * 100
    r2 = bell_score(stop_pct, 2.0, 8.0, falloff=4.0)
    out["R2_stop_distance_pct"] = {"value": stop_pct, "score": r2}

    # R3: stop distance ATR — sweet spot 1.5–4
    if atr14 > 0:
        stop_atr = (price - invalid) / atr14
        r3 = bell_score(stop_atr, 1.5, 4.0, falloff=2.0)
    else:
        stop_atr, r3 = 0.0, 50.0
    out["R3_stop_distance_atr"] = {"value": stop_atr, "score": r3}

    # R4: R/R potential ≥ 2
    if target and price > invalid:
        rr = (target - price) / max(price - invalid, 1e-9)
        r4 = lerp_score(rr, 1.0, 3.0)  # 1R → 0, 3R+ → 100
        out["R4_rr_potential"] = {"value": rr, "score": r4}
    else:
        out["R4_rr_potential"] = {"value": None, "score": 50.0}

    weights = {"R1_stop_clarity": 0.35, "R2_stop_distance_pct": 0.30,
               "R3_stop_distance_atr": 0.15, "R4_rr_potential": 0.20}
    out["_score"] = float(round(sum(out[k]["score"] * w for k, w in weights.items()), 2))
    return out


# ---------------------------------------------------------------------------
# M. Market Context (§6.M)
# ---------------------------------------------------------------------------

def market_context(market_daily: pd.DataFrame | None,
                    vix_daily: pd.DataFrame | None,
                    earnings_days: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # M1: SPY > 50EMA > 200EMA
    if market_daily is not None and len(market_daily) >= 200:
        last = market_daily.iloc[-1]
        if "ema_50" in market_daily.columns and "ema_200" in market_daily.columns:
            stack = (last["close"] > last["ema_50"]) and (last["ema_50"] > last["ema_200"])
            partial = (last["close"] > last["ema_200"])
            m1 = 100.0 if stack else (60.0 if partial else 0.0)
        else:
            m1 = 50.0
    else:
        m1 = 50.0
    out["M1_spy_regime"] = {"score": m1}

    # M3: earnings risk
    if earnings_days is None:
        m3 = 80.0  # neutral-ish if unknown
    elif earnings_days <= 2:
        m3 = 0.0
    elif earnings_days <= 5:
        m3 = 30.0
    elif earnings_days <= 10:
        m3 = 70.0
    else:
        m3 = 100.0
    out["M3_earnings_risk"] = {"value": earnings_days, "score": m3}

    # M4: VIX
    if vix_daily is not None and len(vix_daily) > 0:
        vix = float(vix_daily["close"].iloc[-1])
        m4 = clamp(100 - (vix - 15) / 15 * 100, 0, 100)
    else:
        vix, m4 = None, 60.0
    out["M4_vix"] = {"value": vix, "score": m4}

    # Aggregate (M2 sector RS deferred to v2 — needs sector mapping)
    weights = {"M1_spy_regime": 0.40, "M3_earnings_risk": 0.40, "M4_vix": 0.20}
    out["_score"] = float(round(sum(out[k]["score"] * w for k, w in weights.items()), 2))
    return out


# ---------------------------------------------------------------------------
# L. Liquidity (§6.L)
# ---------------------------------------------------------------------------

def liquidity(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    last = daily.iloc[-1]
    price = float(last["close"])

    # L1: price band
    l1 = bell_score(price, 10.0, 500.0, falloff=200.0)
    out["L1_price"] = {"value": price, "score": l1}

    # L2: dollar volume (log-scaled)
    dv = float(last["close"] * last["volume"])
    if dv >= 100_000_000:
        l2 = 100.0
    elif dv <= 1_000_000:
        l2 = 0.0
    else:
        l2 = (np.log10(dv) - 6) / 2 * 100
    out["L2_dollar_vol"] = {"value": dv, "score": float(l2)}

    # L3: spread proxy (intraday range / price — wide range often means wide spread)
    rng = float((last["high"] - last["low"]) / max(last["close"], 1e-9))
    l3 = clamp(100 - rng * 1000, 0, 100)
    out["L3_spread_proxy"] = {"value": rng, "score": l3}

    # L4: volume z-score (penalize extreme outliers — often news events)
    if len(daily) >= 20:
        v = daily["volume"].tail(20)
        if v.std() > 0:
            z = float((last["volume"] - v.mean()) / v.std())
            if z > 4:
                l4 = 30.0
            elif z > 2:
                l4 = 70.0
            else:
                l4 = 100.0
        else:
            z, l4 = 0.0, 80.0
    else:
        z, l4 = 0.0, 80.0
    out["L4_volume_zscore"] = {"value": z, "score": l4}

    weights = {"L1_price": 0.25, "L2_dollar_vol": 0.40, "L3_spread_proxy": 0.10, "L4_volume_zscore": 0.25}
    out["_score"] = float(round(sum(out[k]["score"] * w for k, w in weights.items()), 2))
    return out
