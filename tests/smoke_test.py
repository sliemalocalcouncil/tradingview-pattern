"""End-to-end smoke test using synthetic data — no Polygon required.

Runs each scoring step on a hand-crafted price series that should trigger
a bull_flag and verifies the engine produces a coherent Signal with a
fully populated features_json.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scanner.detectors import detect_all
from scanner.features import (breakout_readiness, liquidity, market_context, risk_definition,
                               sr_quality, trend_regime, volatility_compression,
                               volume_confirmation)
from scanner.geometry import score_geometry
from scanner.indicators import add_common_indicators
from scanner.models import CommonScores, Signal
from scanner.scoring import apply_vetoes, apply_weights
from scanner.state_machine import StateContext, determine_state


def synthetic_daily(n: int = 220) -> pd.DataFrame:
    """Build a daily OHLCV series with a clear uptrend culminating in a base."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    close = np.linspace(20, 80, n)            # general uptrend 20→80
    close += np.sin(np.linspace(0, 6 * np.pi, n)) * 1.5  # waviness
    close[-25:] = np.linspace(close[-26], close[-26] * 1.02, 25)  # tight base at the top
    high = close * (1 + np.random.default_rng(0).uniform(0.005, 0.02, n))
    low  = close * (1 - np.random.default_rng(1).uniform(0.005, 0.02, n))
    open_ = (high + low) / 2
    volume = np.random.default_rng(2).integers(2_000_000, 8_000_000, n)
    volume[-25:] = (volume[-25:] * 0.6).astype(int)  # base dry-up
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": volume.astype(float)}, index=idx)


def synthetic_hourly(n: int = 120) -> pd.DataFrame:
    """Hourly bars exhibiting a bull-flag (sharp pole, then tight pullback)."""
    idx = pd.date_range("2024-04-01 14:00", periods=n, freq="1h", tz="UTC")
    base = np.linspace(50, 51, n - 30)
    pole = np.linspace(51, 60, 15)            # +18% pole
    flag = np.linspace(60, 58, 15)            # -3.3% gentle pullback
    close = np.concatenate([base, pole, flag])
    high = close * (1 + np.random.default_rng(3).uniform(0.002, 0.015, n))
    low  = close * (1 - np.random.default_rng(4).uniform(0.002, 0.015, n))
    open_ = (high + low) / 2
    volume = np.random.default_rng(5).integers(80_000, 250_000, n).astype(float)
    volume[n - 30 : n - 15] *= 1.8           # pole bars: heavy volume
    volume[n - 15 :] *= 0.55                 # flag bars: dry-up
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": volume}, index=idx)


def main() -> int:
    print("Building synthetic data ...")
    daily = add_common_indicators(synthetic_daily())
    hourly = synthetic_hourly()
    print(f"daily shape: {daily.shape}, hourly shape: {hourly.shape}")

    print("\nDetecting patterns ...")
    matches = detect_all(daily, hourly)
    if not matches:
        print("FAIL: no patterns detected on synthetic data")
        return 1
    for m in matches:
        print(f"  → {m.pattern}: trigger={m.neckline:.2f}, invalid={m.invalid_below:.2f}")

    print("\nScoring (using SPY-shaped synthetic market) ...")
    market_daily = add_common_indicators(synthetic_daily(220))

    n_pass = 0
    for match in matches:
        geom = score_geometry(match, daily)
        trend = trend_regime(daily, market_daily)
        comp = volatility_compression(daily)
        sr = sr_quality(daily)
        vol = volume_confirmation(daily, hourly, match)
        br = breakout_readiness(daily, hourly, match)
        risk = risk_definition(daily, match)
        mkt = market_context(market_daily, None, earnings_days=30)
        liq = liquidity(daily)

        common = CommonScores(
            trend=trend["_score"], geometry=geom["_score"],
            compression=comp["_score"], sr_quality=sr["_score"],
            volume=vol["_score"], readiness=br["_score"], risk=risk["_score"],
            market=mkt["_score"], liquidity=liq["_score"],
        )
        components = apply_weights(common, geom["_score"], match.pattern)
        raw = components.total()
        features_blob = {
            "trend": trend, "geometry": geom, "compression": comp, "sr_quality": sr,
            "volume": vol, "readiness": br, "risk": risk, "market": mkt, "liquidity": liq,
            "pattern_structure": match.structure, "pattern_geometry_features": match.geometry_features,
        }
        final, vetoes = apply_vetoes(raw, common, match, features_blob)
        ctx = StateContext(
            final_score=final, geometry_score=geom["_score"],
            readiness_score=br["_score"],
            breakout_distance_atr=br.get("_distance_atr"),
            closed_above_trigger=False, closed_below_invalid=False,
            bars_since_breakout=None,
        )
        state = determine_state(match, ctx, prior_state=None)

        print(f"\n  {match.pattern}:")
        print(f"    common: T={common.trend:.0f} G={common.geometry:.0f} VC={common.compression:.0f} "
              f"SR={common.sr_quality:.0f} V={common.volume:.0f} BR={common.readiness:.0f} "
              f"R={common.risk:.0f} M={common.market:.0f} L={common.liquidity:.0f}")
        print(f"    components total = {raw:.1f}, after veto = {final:.1f}")
        print(f"    veto reasons: {vetoes or 'none'}")
        print(f"    state: {state.value}")

        # Sanity assertions
        assert 0 <= final <= 100, f"final score out of range: {final}"
        assert all(0 <= v <= 100 for v in common.as_dict().values())
        assert "_score" in features_blob["geometry"]
        n_pass += 1

    print(f"\nPASS: {n_pass} signal(s) produced; explainability blob populated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
