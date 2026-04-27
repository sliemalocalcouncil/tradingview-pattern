"""Pattern state machine — per design doc §7.6.

States:
    absent → forming → candidate → setup → breakout_confirmed
                                              ↘ retest_hold ↺
                                              ↘ extended (terminal)
                                              ↘ breakout_failed → absent
                                              ↘ invalidated → absent

The state determines the alert type the alerts module produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import PatternMatch


class PatternState(str, Enum):
    ABSENT = "absent"
    FORMING = "forming"
    CANDIDATE = "candidate"
    SETUP = "setup"
    BREAKOUT_CONFIRMED = "breakout_confirmed"
    RETEST_HOLD = "retest_hold"
    EXTENDED = "extended"
    BREAKOUT_FAILED = "breakout_failed"
    INVALIDATED = "invalidated"


# A "signaling" state is one we'd ever consider sending an alert about.
SIGNALING_STATES = {
    PatternState.SETUP, PatternState.BREAKOUT_CONFIRMED, PatternState.RETEST_HOLD,
    PatternState.BREAKOUT_FAILED, PatternState.INVALIDATED,
}

NEAR_BREAKOUT_STATES = {PatternState.SETUP, PatternState.CANDIDATE}

# States that should be tracked by the active-signal monitor even after the
# detector stops returning a match. Per review §P0-2.
ACTIVE_MONITORING_STATES = {
    PatternState.SETUP, PatternState.BREAKOUT_CONFIRMED, PatternState.RETEST_HOLD,
}


@dataclass
class StateContext:
    final_score: float
    geometry_score: float
    readiness_score: float
    breakout_distance_atr: float | None  # negative if already broken out
    closed_above_trigger: bool
    closed_below_invalid: bool
    bars_since_breakout: int | None
    # Review §P1-b: require a volume bar for BREAKOUT_CONFIRMED promotion.
    # Default True keeps behavior backward-compatible for tests / synthetic data.
    volume_confirmed: bool = True


def determine_state(match: PatternMatch, ctx: StateContext, prior_state: PatternState | None,
                    min_candidate: float = 60.0) -> PatternState:
    """Compute the new state given the latest signal context and prior state."""

    # Hard exits first
    if ctx.closed_below_invalid:
        return PatternState.INVALIDATED

    # If price extended way above trigger
    if ctx.bars_since_breakout is not None and ctx.breakout_distance_atr is not None \
            and ctx.breakout_distance_atr < -3.0:
        return PatternState.EXTENDED

    # Failed breakout: previously confirmed, now closed back below trigger
    # (review §P0-1: this branch was dead code without prior_state)
    if prior_state == PatternState.BREAKOUT_CONFIRMED and not ctx.closed_above_trigger:
        return PatternState.BREAKOUT_FAILED

    # Closed above trigger
    if ctx.closed_above_trigger:
        # Retest after prior confirmation: pulled back to / through trigger and held
        if prior_state in (PatternState.BREAKOUT_CONFIRMED, PatternState.RETEST_HOLD):
            if ctx.breakout_distance_atr is not None and -0.3 < ctx.breakout_distance_atr <= 0.5:
                return PatternState.RETEST_HOLD
        # Promote to confirmed only with volume confirmation (review §P1-b)
        if ctx.volume_confirmed:
            return PatternState.BREAKOUT_CONFIRMED
        # Closed above without volume: hold prior confirmed state if any, else SETUP
        if prior_state == PatternState.BREAKOUT_CONFIRMED:
            return PatternState.BREAKOUT_CONFIRMED
        return PatternState.SETUP

    # Setup vs candidate vs forming
    if ctx.final_score >= 75 and ctx.readiness_score >= 70:
        return PatternState.SETUP
    if ctx.final_score >= min_candidate:
        return PatternState.CANDIDATE
    if ctx.geometry_score >= 50:
        return PatternState.FORMING
    return PatternState.ABSENT
