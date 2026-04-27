"""Per-pattern smoke test for the 5 new detectors.

Builds synthetic OHLCV series tuned to each pattern's geometry and verifies:
  - the corresponding detector fires,
  - the geometry scorer returns a sensible score,
  - the structure / geometry_features blobs are populated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scanner.detectors import (detect_base_on_base, detect_breakout_retest_hold,
                                 detect_cup_with_handle, detect_inverse_head_shoulders,
                                 detect_tight_consolidation)
from scanner.geometry import score_geometry
from scanner.indicators import add_common_indicators


def _ohlc_from_close(close: np.ndarray, vol: np.ndarray, idx: pd.DatetimeIndex,
                      noise_scale: float = 0.005, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    high = close * (1 + rng.uniform(noise_scale, noise_scale * 3, len(close)))
    low = close * (1 - rng.uniform(noise_scale, noise_scale * 3, len(close)))
    open_ = (high + low) / 2
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": vol.astype(float)}, index=idx)


def make_cup_with_handle() -> pd.DataFrame:
    """Daily series: long uptrend → cup (U) → handle pullback → ready."""
    idx = pd.date_range("2024-01-01", periods=220, freq="B", tz="UTC")
    # Pre-base run-up to ~100
    pre = np.linspace(60, 100, 100)
    # Cup: 100 → 80 → 100 over 60 bars (U-shape)
    cup_x = np.linspace(0, np.pi, 60)
    cup = 100 - 20 * np.sin(cup_x)
    # Handle: 100 → 95 → 96 over 12 bars (shallow pullback)
    handle = np.array([100, 99, 98.5, 98, 97.5, 97, 96.8, 96.5, 96.2, 96.0, 95.8, 96])
    # Final 48 bars near the rim, gently rising
    after = np.linspace(96, 97, 48)
    close = np.concatenate([pre, cup, handle, after])
    assert len(close) == 220
    vol = np.random.default_rng(11).integers(2_000_000, 8_000_000, 220).astype(float)
    vol[160:172] *= 0.55  # handle dry-up
    return _ohlc_from_close(close, vol, idx, seed=11)


def make_inverse_head_shoulders() -> pd.DataFrame:
    """Hourly series with three clear lows: shoulder, deeper head, shoulder."""
    n = 100
    idx = pd.date_range("2024-04-01 14:00", periods=n, freq="1h", tz="UTC")
    # Build base around 100 with three depressions
    base = np.full(n, 100.0)
    # Left shoulder around bar 20: dip to 95
    for i in range(15, 26):
        base[i] = 100 - 5 * np.sin(np.pi * (i - 15) / 11)
    # Head around bar 50: dip to 90 (deeper)
    for i in range(43, 58):
        base[i] = 100 - 10 * np.sin(np.pi * (i - 43) / 15)
    # Right shoulder around bar 80: dip to 95.5
    for i in range(73, 84):
        base[i] = 100 - 5 * np.sin(np.pi * (i - 73) / 11)
    # Recovery in last 12 bars
    base[88:] = np.linspace(99, 99.5, n - 88)
    vol = np.random.default_rng(12).integers(80_000, 250_000, n).astype(float)
    return _ohlc_from_close(base, vol, idx, noise_scale=0.003, seed=12)


def make_base_on_base() -> pd.DataFrame:
    """Daily series with two stacked bases, ending in a small pullback so the
    second base's right rim registers as a confirmed pivot high."""
    idx = pd.date_range("2024-01-01", periods=200, freq="B", tz="UTC")
    # Pre-rally to 80
    pre = np.linspace(60, 80, 50)
    # First base: peak 80 → trough 70 → peak 80 over 35 bars (cup-like)
    fb_x = np.linspace(0, np.pi, 35)
    fb = 80 - 10 * np.sin(fb_x)
    # Mini-pullback after first base (creates clean pivot at first_top)
    fb_pullback = np.linspace(80, 78, 5)
    # Advance: 78 → 95
    advance = np.linspace(78, 95, 25)
    # Second base: 95 → 89 → 95 over 35 bars (shallower, low above first_top × 0.93)
    sb_x = np.linspace(0, np.pi, 35)
    sb = 95 - 6 * np.sin(sb_x)
    # Final pullback so second peak registers as pivot
    tail = np.linspace(95, 93, 200 - 50 - 35 - 5 - 25 - 35)
    close = np.concatenate([pre, fb, fb_pullback, advance, sb, tail])
    assert len(close) == 200
    vol = np.random.default_rng(13).integers(2_000_000, 8_000_000, 200).astype(float)
    # Dry-up during second base (115..150)
    vol[115:150] = (vol[115:150] * 0.6).astype(float)
    return _ohlc_from_close(close, vol, idx, seed=13)


def make_tight_consolidation() -> pd.DataFrame:
    """Daily series ending with a tight 14-day box."""
    idx = pd.date_range("2024-01-01", periods=160, freq="B", tz="UTC")
    pre = np.linspace(40, 80, 146)
    box = 80 + np.random.default_rng(14).uniform(-1.5, 1.5, 14)
    close = np.concatenate([pre, box])
    vol = np.random.default_rng(15).integers(2_000_000, 8_000_000, 160).astype(float)
    vol[146:] = (vol[146:] * 0.6).astype(float)  # dry up during the box
    return _ohlc_from_close(close, vol, idx, noise_scale=0.003, seed=14)


def make_retest_hold() -> pd.DataFrame:
    """Hourly: build a swing high → consolidate → break above → pull back to the
    level → hold above. Final hold is short so the retest is recent."""
    n = 105
    idx = pd.date_range("2024-04-01 14:00", periods=n, freq="1h", tz="UTC")
    # 0..49: rally to 100
    up1 = np.linspace(85, 100, 50)
    # 50..54: small pullback to 98 (so 100 registers as a swing high pivot)
    pb1 = np.linspace(100, 98, 5)
    # 55..64: re-approach 100
    reapproach = np.linspace(98, 100, 10)
    # 65..79: breakout from 100 to 104
    bo = np.linspace(100, 104, 15)
    # 80..94: pullback (retest) from 104 to 100.5
    pb2 = np.linspace(104, 100.5, 15)
    # 95..104: short hold, ending with a small dip so we have valid trailing pivots
    hold = np.concatenate([
        np.linspace(100.5, 102.5, 6),
        np.linspace(102.5, 101.8, 4),
    ])
    close = np.concatenate([up1, pb1, reapproach, bo, pb2, hold])
    assert len(close) == n
    vol = np.random.default_rng(16).integers(80_000, 250_000, n).astype(float)
    vol[65:80] = (vol[65:80] * 1.6).astype(float)  # heavy on breakout
    return _ohlc_from_close(close, vol, idx, noise_scale=0.003, seed=16)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def main() -> int:
    results: list[tuple[str, bool, str]] = []

    # 1. Cup with Handle
    daily = add_common_indicators(make_cup_with_handle())
    m = detect_cup_with_handle(daily)
    if m is None:
        results.append(("cup_with_handle", False, "detector returned None"))
    else:
        g = score_geometry(m, daily)
        ok = 0 <= g["_score"] <= 100 and m.neckline > 0
        results.append(("cup_with_handle", ok,
                         f"trigger={m.neckline:.2f}, geo={g['_score']:.1f}, "
                         f"depth={m.geometry_features['cup_depth_pct']:.2%}"))

    # 2. Inverse H&S
    hourly = make_inverse_head_shoulders()
    daily_for_ihs = add_common_indicators(make_cup_with_handle())  # any daily for ATR
    m = detect_inverse_head_shoulders(hourly, daily_for_ihs)
    if m is None:
        results.append(("inverse_head_shoulders", False, "detector returned None"))
    else:
        g = score_geometry(m, daily_for_ihs)
        ok = 0 <= g["_score"] <= 100 and m.neckline > 0
        results.append(("inverse_head_shoulders", ok,
                         f"trigger={m.neckline:.2f}, geo={g['_score']:.1f}, "
                         f"head_depth={m.geometry_features['head_depth_pct']:.2%}"))

    # 3. Base-on-Base
    daily = add_common_indicators(make_base_on_base())
    m = detect_base_on_base(daily)
    if m is None:
        results.append(("base_on_base", False, "detector returned None"))
    else:
        g = score_geometry(m, daily)
        ok = 0 <= g["_score"] <= 100 and m.neckline > 0
        results.append(("base_on_base", ok,
                         f"trigger={m.neckline:.2f}, geo={g['_score']:.1f}, "
                         f"first_depth={m.geometry_features['first_base_depth_pct']:.2%}, "
                         f"second_depth={m.geometry_features['second_base_depth_pct']:.2%}"))

    # 4. Tight Consolidation
    daily = add_common_indicators(make_tight_consolidation())
    m = detect_tight_consolidation(daily)
    if m is None:
        results.append(("tight_consolidation", False, "detector returned None"))
    else:
        g = score_geometry(m, daily)
        ok = 0 <= g["_score"] <= 100 and m.neckline > 0
        results.append(("tight_consolidation", ok,
                         f"trigger={m.neckline:.2f}, geo={g['_score']:.1f}, "
                         f"window={int(m.geometry_features['duration_bars'])}, "
                         f"range_atr={m.geometry_features['range_atr']:.2f}"))

    # 5. Retest Hold
    hourly = make_retest_hold()
    daily_for_rh = add_common_indicators(make_cup_with_handle())
    m = detect_breakout_retest_hold(hourly, daily_for_rh)
    if m is None:
        results.append(("breakout_retest_hold", False, "detector returned None"))
    else:
        g = score_geometry(m, daily_for_rh)
        ok = 0 <= g["_score"] <= 100 and m.neckline > 0
        results.append(("breakout_retest_hold", ok,
                         f"trigger={m.neckline:.2f}, geo={g['_score']:.1f}, "
                         f"retest_atr={m.geometry_features['retest_depth_atr']:.2f}"))

    # Print results
    print("\n" + "=" * 70)
    print(f"{'Pattern':<28} {'Pass':<6} Notes")
    print("=" * 70)
    n_pass = 0
    for name, ok, notes in results:
        flag = "✓" if ok else "✗"
        print(f"{name:<28} {flag:<6} {notes}")
        if ok:
            n_pass += 1
    print("=" * 70)
    print(f"\n{n_pass}/{len(results)} pattern detectors verified")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
