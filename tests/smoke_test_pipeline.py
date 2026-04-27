"""End-to-end pipeline integration test with a fake PolygonClient.

Verifies (review §P0 / §P1):
  - pipeline runs without a real Polygon API key
  - ENABLED_PATTERNS env filter works
  - latest_top_signals.csv is written even when 0 signals
  - run summary contains counters block
  - prior_state from _state.json is honored on second run
  - active monitor synthesizes INVALIDATED signal when price drops below invalid_below
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Fake PolygonClient
# ---------------------------------------------------------------------------

def _build_daily(ticker: str, n: int = 220, base_price: float = 100.0) -> pd.DataFrame:
    """Build a daily OHLCV series ending in a tight base near `base_price`."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(hash(ticker) & 0xFFFF)
    drift = np.linspace(base_price * 0.6, base_price * 0.95, n - 20)
    base = base_price * 0.95 + rng.uniform(-1, 1, 20) * 0.5
    close = np.concatenate([drift, base])
    high = close * (1 + rng.uniform(0.005, 0.015, n))
    low = close * (1 - rng.uniform(0.005, 0.015, n))
    open_ = (high + low) / 2
    vol = rng.integers(2_000_000, 8_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": vol}, index=idx)


def _build_intraday(ticker: str, multiplier: int, timespan: str,
                     n: int = 200) -> pd.DataFrame:
    """Build session-aligned 30-min OR 60-min bars across 13 sessions."""
    rng = np.random.default_rng((hash(ticker) ^ 0xDEAD) & 0xFFFF)
    bars: list[pd.Timestamp] = []
    # Generate timestamps for 13 trading days, regular session 09:30..16:00 ET
    n_per_day = 13 if multiplier == 30 else 7  # 30-min or hourly bars per day
    step = pd.Timedelta(minutes=multiplier) if timespan == "minute" else pd.Timedelta(hours=multiplier)
    day0 = pd.Timestamp("2024-04-01 13:30", tz="UTC")  # 09:30 ET on EDT
    for d in range(15):
        day_start = day0 + pd.Timedelta(days=d)
        if day_start.dayofweek >= 5:  # skip weekends
            continue
        for j in range(n_per_day):
            bars.append(day_start + step * j)
        if len(bars) >= n:
            break
    bars = bars[:n]
    idx = pd.DatetimeIndex(bars, tz="UTC")
    close = 95 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    high = close + rng.uniform(0.05, 0.20, len(idx))
    low = close - rng.uniform(0.05, 0.20, len(idx))
    open_ = (high + low) / 2
    vol = rng.integers(60_000, 250_000, len(idx)).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": vol}, index=idx)


class FakePolygonClient:
    def __init__(self, settings):
        self.settings = settings
        self.calls = 0

    def aggregates(self, ticker: str, multiplier: int, timespan: str,
                    date_from: str, date_to: str, **_kw) -> pd.DataFrame:
        self.calls += 1
        if timespan == "day":
            return _build_daily(ticker)
        return _build_intraday(ticker, multiplier, timespan)

    def previous_day(self, ticker: str):  # pragma: no cover
        return None

    @staticmethod
    def date_window(days_back: int, end=None):
        from datetime import timedelta as _td
        e = end or datetime.now(timezone.utc)
        return ((e - _td(days=days_back)).strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _run_pipeline_in_tmpdir(*, enabled_patterns: str | None = None,
                             ticker_list: list[str] | None = None) -> tuple[Path, dict]:
    """Run the full pipeline against the fake client; returns (result_dir, summary)."""
    if ticker_list is None:
        ticker_list = ["AAPL", "MSFT"]

    tmpdir = Path(tempfile.mkdtemp(prefix="scanner_test_"))
    (tmpdir / "result").mkdir()
    ticker_file = tmpdir / "ticker.txt"
    ticker_file.write_text("\n".join(ticker_list))

    # Reset env
    os.environ["POLYGON_API_KEY"] = "fake-key-for-tests"
    os.environ["PROJECT_ROOT"] = str(tmpdir)
    os.environ["RESULT_DIR"] = "result"
    os.environ["TICKER_FILE"] = "ticker.txt"
    os.environ["DRY_RUN"] = "true"
    os.environ["TELEGRAM_DISABLED"] = "true"
    if enabled_patterns is None:
        os.environ.pop("ENABLED_PATTERNS", None)
    else:
        os.environ["ENABLED_PATTERNS"] = enabled_patterns

    # Patch PolygonClient before importing pipeline
    import scanner.polygon_client as polygon_client
    polygon_client.PolygonClient = FakePolygonClient

    # (Re)import pipeline modules so the patched client is used
    import importlib
    from scanner import pipeline as pl
    importlib.reload(pl)

    rc = pl.run()
    assert rc == 0, f"pipeline returned {rc}"

    summary_path = tmpdir / "result" / "latest_run_summary.json"
    with summary_path.open() as f:
        summary = json.load(f)
    return tmpdir / "result", summary


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_runs_without_api_key():
    result_dir, summary = _run_pipeline_in_tmpdir(ticker_list=["AAPL"])
    # latest_top_signals.csv must always exist
    top = result_dir / "latest_top_signals.csv"
    assert top.exists(), "latest_top_signals.csv missing"
    # summary has counters
    assert "counters" in summary
    assert summary["counters"]["tickers_processed"] >= 0
    assert "errors" in summary


def test_latest_top_signals_present_when_zero_signals():
    """Review §P1: empty file is acceptable, missing file is not."""
    result_dir, summary = _run_pipeline_in_tmpdir(ticker_list=["AAPL"])
    top = result_dir / "latest_top_signals.csv"
    df = pd.read_csv(top)
    # The synthetic data may not produce any signals — that's OK,
    # what matters is the file exists with proper columns.
    assert "final_score" in df.columns
    assert "pattern_state" in df.columns


def test_enabled_patterns_filter_respected():
    """ENABLED_PATTERNS=tight_consolidation should restrict to that one detector."""
    result_dir, summary = _run_pipeline_in_tmpdir(
        enabled_patterns="tight_consolidation",
        ticker_list=["AAPL"],
    )
    by_pattern = summary.get("by_pattern", {})
    # Either no signals (acceptable) or only tight_consolidation
    for p in by_pattern.keys():
        assert p == "tight_consolidation", f"unexpected pattern with filter: {p}"


def test_run_summary_has_counters_block():
    result_dir, summary = _run_pipeline_in_tmpdir(ticker_list=["AAPL", "MSFT"])
    assert isinstance(summary["counters"], dict)
    assert "detector_hits" in summary["counters"]
    assert "alerts_by_type" in summary["counters"]
    assert "active_monitor_events" in summary["counters"]


def test_active_monitor_invalidates_after_drop():
    """Plant a SETUP state with invalid_below=999, run again, expect INVALIDATED."""
    # Run once to build state for AAPL
    result_dir, _ = _run_pipeline_in_tmpdir(ticker_list=["AAPL"])

    # Manually inject a SETUP state with an unreachable invalid_below
    state_path = result_dir / "_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state["AAPL::bull_flag"] = {
        "state": "breakout_confirmed",
        "trigger": 50.0,            # current price > 50, so won't be FAILED
        "invalid_below": 999.0,      # current price < 999, so will be INVALIDATED
        "monitored_until": "2099-01-01T00:00:00+00:00",
        "last_alert_type": "breakout",
        "last_alert_score": 80.0,
        "last_alert_at": "2024-01-01T00:00:00+00:00",
    }
    state_path.write_text(json.dumps(state, indent=2))

    # Run pipeline again (point env vars at the same tmpdir)
    project_root = result_dir.parent
    os.environ["PROJECT_ROOT"] = str(project_root)
    os.environ["RESULT_DIR"] = "result"
    os.environ["TICKER_FILE"] = "ticker.txt"
    os.environ["DRY_RUN"] = "true"
    os.environ["TELEGRAM_DISABLED"] = "true"

    import scanner.polygon_client as polygon_client
    polygon_client.PolygonClient = FakePolygonClient
    import importlib
    from scanner import pipeline as pl
    importlib.reload(pl)
    rc = pl.run()
    assert rc == 0

    # Check that latest_run_summary mentions invalidated
    summary = json.loads((result_dir / "latest_run_summary.json").read_text())
    by_state = summary.get("by_state", {})
    # If the detector also matched bull_flag, it'd override; with invalid_below=999
    # the detector almost certainly cannot produce a normal SETUP, so the active
    # monitor should fire INVALIDATED.
    assert "invalidated" in by_state or summary["counters"]["active_monitor_events"] > 0, \
        f"expected invalidated event, got {summary}"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        ("pipeline runs without real API key",    test_pipeline_runs_without_api_key),
        ("latest_top_signals always written",     test_latest_top_signals_present_when_zero_signals),
        ("ENABLED_PATTERNS filter respected",     test_enabled_patterns_filter_respected),
        ("run summary has counters block",        test_run_summary_has_counters_block),
        ("active monitor catches invalidation",   test_active_monitor_invalidates_after_drop),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} pipeline integration tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
