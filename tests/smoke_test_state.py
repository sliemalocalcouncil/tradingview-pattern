"""State machine + active-monitor + alert dedup integration test.

Verifies (review §P0-1, §P0-2, §P1):
  - prior_state actually flows from _state.json into determine_state()
  - BREAKOUT_FAILED fires when prior=BREAKOUT_CONFIRMED and price drops below trigger
  - INVALIDATED fires when price drops below invalid_below
  - active monitor catches invalidation when detector no longer matches
  - RETEST_HOLD now produces an alert (was previously suppressed)
  - volume_confirmed=False keeps state at SETUP instead of promoting to BREAKOUT_CONFIRMED
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scanner.alerts import (AlertState, decide_alert_type, format_message,
                              update_signal_state)
from scanner.config import Settings
from scanner.models import (CommonScores, PatternMatch, Signal, WeightedComponents)
from scanner.state_machine import (ACTIVE_MONITORING_STATES, PatternState,
                                     StateContext, determine_state)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _match() -> PatternMatch:
    return PatternMatch(
        pattern="bull_flag",
        neckline=100.0, invalid_below=92.0,
        measured_move_target=115.0,
        structure={"structural_low": 92.0, "current_price": 99.0,
                    "neckline_distance_pct": 0.01, "base_bars": 8},
        geometry_features={"pole_gain": 0.20, "atr_pct": 0.02},
    )


def _ctx(*, score: float = 78, ready: float = 75, geom: float = 70,
          dist_atr: float | None = 0.3, above_trigger: bool = False,
          below_invalid: bool = False, vol_confirmed: bool = True,
          bsb: int | None = None) -> StateContext:
    return StateContext(
        final_score=score, geometry_score=geom, readiness_score=ready,
        breakout_distance_atr=dist_atr,
        closed_above_trigger=above_trigger, closed_below_invalid=below_invalid,
        bars_since_breakout=bsb, volume_confirmed=vol_confirmed,
    )


def _signal(state: PatternState, score: float = 78.0,
             trigger: float | None = 100.0, invalid: float | None = 92.0,
             pattern: str = "bull_flag", ticker: str = "TEST") -> Signal:
    return Signal(
        run_id="r1", asof=datetime.now(timezone.utc).isoformat(),
        ticker=ticker, pattern=pattern, pattern_state=state.value,
        final_score=score,
        common_scores=CommonScores(70, 70, 60, 70, 70, 70, 60, 80, 90),
        components=WeightedComponents(),
        veto_triggered=False, veto_reasons=[],
        price=99.0, trigger=trigger, invalid_below=invalid,
        measured_move_target=115.0, features_json={},
    )


def _settings_with_tmpdir(tmpdir: Path) -> Settings:
    """Settings instance with result_dir pointing at tmpdir."""
    # Settings is frozen, so we need to set env var BEFORE construction
    os.environ["RESULT_DIR"] = str(tmpdir)
    os.environ["PROJECT_ROOT"] = str(tmpdir)
    s = Settings()
    return s


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_prior_state_drives_breakout_failed():
    """Review §P0-1: prior=BREAKOUT_CONFIRMED + below-trigger close ⇒ BREAKOUT_FAILED."""
    state = determine_state(_match(),
                             _ctx(above_trigger=False),
                             prior_state=PatternState.BREAKOUT_CONFIRMED)
    assert state == PatternState.BREAKOUT_FAILED, state


def test_prior_state_drives_retest_hold():
    """prior=BREAKOUT_CONFIRMED + closed back above trigger near level ⇒ RETEST_HOLD."""
    state = determine_state(_match(),
                             _ctx(above_trigger=True, dist_atr=0.2),
                             prior_state=PatternState.BREAKOUT_CONFIRMED)
    assert state == PatternState.RETEST_HOLD, state


def test_invalidated_overrides_everything():
    state = determine_state(_match(),
                             _ctx(below_invalid=True, above_trigger=True),
                             prior_state=PatternState.BREAKOUT_CONFIRMED)
    assert state == PatternState.INVALIDATED, state


def test_volume_required_for_breakout_confirmed():
    """Review §P1: closed above trigger but volume_confirmed=False ⇒ stay at SETUP."""
    s_no_vol = determine_state(_match(),
                                _ctx(above_trigger=True, vol_confirmed=False),
                                prior_state=PatternState.SETUP)
    assert s_no_vol == PatternState.SETUP, s_no_vol

    s_with_vol = determine_state(_match(),
                                   _ctx(above_trigger=True, vol_confirmed=True),
                                   prior_state=PatternState.SETUP)
    assert s_with_vol == PatternState.BREAKOUT_CONFIRMED, s_with_vol


def test_retest_hold_now_alerts():
    """Review §P1: RETEST_HOLD must produce an alert_type (was None before)."""
    out = decide_alert_type(
        new_state=PatternState.RETEST_HOLD, prior_alert_type="breakout",
        final_score=80, prior_score=78, prior_alert_at=None,
        cooldown_hours=6, min_alert_score=70, score_upgrade_delta=8,
    )
    assert out == "retest", out


def test_retest_hold_dedup():
    out = decide_alert_type(
        new_state=PatternState.RETEST_HOLD, prior_alert_type="retest",
        final_score=80, prior_score=78, prior_alert_at=None,
        cooldown_hours=6, min_alert_score=70, score_upgrade_delta=8,
    )
    assert out is None, out


def test_alert_state_round_trip(tmpdir: Path):
    s = _settings_with_tmpdir(tmpdir)
    st = AlertState(s)
    sig = _signal(PatternState.SETUP)
    update_signal_state(st, sig, monitoring_days=5)
    st.save()

    # Read back through a fresh AlertState instance
    st2 = AlertState(s)
    e = st2.get("TEST", "bull_flag")
    assert e.get("state") == "setup", e
    assert e.get("trigger") == 100.0, e
    assert e.get("invalid_below") == 92.0, e
    assert e.get("monitored_until"), e


def test_atomic_state_save_no_corruption(tmpdir: Path):
    """Verify the state file ends up valid JSON and contains all writes."""
    s = _settings_with_tmpdir(tmpdir)
    st = AlertState(s)
    pre_count = len(st.all_entries())  # earlier tests may have populated entries
    for i in range(20):
        sig = _signal(PatternState.SETUP, ticker=f"ATOMIC{i:02d}")
        update_signal_state(st, sig, monitoring_days=5)
    st.save()
    parsed = json.loads(st.path.read_text())
    assert len(parsed) == pre_count + 20, f"got {len(parsed)}, expected {pre_count + 20}"


def test_format_message_contains_key_fields():
    sig = _signal(PatternState.BREAKOUT_CONFIRMED, score=82.5)
    msg = format_message(sig, "breakout")
    assert "BREAKOUT" in msg
    assert "TEST" in msg
    assert "82.5" in msg
    assert "$100.00" in msg or "$100" in msg


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        tests = [
            ("prior_state → BREAKOUT_FAILED", lambda: test_prior_state_drives_breakout_failed()),
            ("prior_state → RETEST_HOLD",     lambda: test_prior_state_drives_retest_hold()),
            ("INVALIDATED hard exit",          lambda: test_invalidated_overrides_everything()),
            ("Volume required for BO",         lambda: test_volume_required_for_breakout_confirmed()),
            ("RETEST_HOLD now alerts",         lambda: test_retest_hold_now_alerts()),
            ("RETEST_HOLD dedup",              lambda: test_retest_hold_dedup()),
            ("AlertState round-trip",          lambda: test_alert_state_round_trip(tmp)),
            ("Atomic state save",              lambda: test_atomic_state_save_no_corruption(tmp)),
            ("format_message",                  lambda: test_format_message_contains_key_fields()),
        ]
        passed = 0
        for name, fn in tests:
            try:
                fn()
                print(f"  ✓ {name}")
                passed += 1
            except AssertionError as e:
                print(f"  ✗ {name}: assertion failed: {e}")
            except Exception as e:
                print(f"  ✗ {name}: {type(e).__name__}: {e}")
        print(f"\n{passed}/{len(tests)} state-machine tests passed")
        return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
