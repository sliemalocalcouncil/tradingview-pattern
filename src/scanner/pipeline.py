"""Main scanning pipeline.

Per-ticker flow:
    fetch daily + ET-aligned hourly bars
        → indicators
        → detect patterns (configurable enabled_patterns)
        → for each match:
            common features (T, VC, SR, V, BR, R, M, L)
            pattern geometry (G)
            apply weight matrix → raw score
            apply veto rules → final score + reasons
            determine pattern state (state machine, fed prior_state from _state.json)
        → save state for ALL signals (so active monitor can find them next cycle)
        → maybe alert ONLY for the best signal per ticker (deduplication)
        → for any active signal not in current matches, monitor invalidation/failure
        → persist signal row to CSV + JSON
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .alerts import AlertState, maybe_alert, update_signal_state
from .config import Settings, load_settings
from .data import (fetch_daily_bars, fetch_native_hourly,
                    fetch_session_aligned_hourly,
                    latest_complete_hourly, to_regular_session_hourly)
from .detectors import detect_all
from .features import (breakout_readiness, liquidity, market_context, risk_definition,
                        sr_quality, trend_regime, volatility_compression, volume_confirmation)
from .geometry import score_geometry
from .indicators import add_common_indicators
from .models import CommonScores, Signal, WeightedComponents
from .polygon_client import PolygonClient, PolygonError
from .scoring import apply_vetoes, apply_weights
from .state_machine import (ACTIVE_MONITORING_STATES, PatternState, StateContext,
                              determine_state)

log = logging.getLogger(__name__)

# Signals that earn an alert decision (others have state saved but no alert).
ALERTABLE_STATES = {
    PatternState.SETUP.value,
    PatternState.BREAKOUT_CONFIRMED.value,
    PatternState.RETEST_HOLD.value,
    PatternState.BREAKOUT_FAILED.value,
    PatternState.INVALIDATED.value,
    PatternState.CANDIDATE.value,
}


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def load_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"ticker file not found: {path}")
    out: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        sym = s.split()[0].upper()
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


# ---------------------------------------------------------------------------
# Per-ticker scan
# ---------------------------------------------------------------------------

@dataclass
class MarketContext:
    spy_daily: pd.DataFrame | None = None
    vix_daily: pd.DataFrame | None = None
    earnings_days_by_ticker: dict[str, int] | None = None


def fetch_market_context(client: PolygonClient, settings: Settings) -> MarketContext:
    """Fetch SPY + VIX once per run; earnings calendar TODO (v2)."""
    ctx = MarketContext()
    try:
        spy = fetch_daily_bars(client, "SPY", settings.daily_lookback_days)
        if not spy.empty:
            ctx.spy_daily = add_common_indicators(spy)
    except PolygonError as e:
        log.warning("SPY fetch failed: %s", e)
    try:
        vix = fetch_daily_bars(client, "VIX", 90)
        if not vix.empty:
            ctx.vix_daily = vix
    except PolygonError:
        log.debug("VIX not available; market_context.M4 will use neutral score")
    return ctx


def _fetch_hourly(client: PolygonClient, settings: Settings, ticker: str,
                   asof: datetime) -> pd.DataFrame:
    """Fetch and align hourly bars per HOURLY_ALIGNMENT_MODE (review §P0-3)."""
    if settings.hourly_alignment_mode == "native":
        raw = fetch_native_hourly(client, ticker, settings.intraday_lookback_days)
        if raw.empty:
            return raw
        return latest_complete_hourly(to_regular_session_hourly(raw, asof=asof))
    # Default: ET-aligned 60-min bars (built from 30-min bars)
    aligned = fetch_session_aligned_hourly(client, ticker, settings.intraday_lookback_days, asof=asof)
    if aligned.empty:
        return aligned
    return latest_complete_hourly(aligned)


def _is_volume_confirmed(features_blob: dict[str, Any], threshold: float = 60.0) -> bool:
    """Review §P1: a true breakout needs an expansion volume bar."""
    vol_blob = features_blob.get("volume", {})
    v3 = vol_blob.get("V3_breakout_bar", {}).get("score", 0)
    v5 = vol_blob.get("V5_bo_expansion", {}).get("score", 0)
    return float(max(v3, v5)) >= threshold


def scan_ticker(client: PolygonClient, settings: Settings, ticker: str,
                run_id: str, asof: datetime, market_ctx: MarketContext,
                alert_state: AlertState) -> tuple[list[Signal], list[Signal]]:
    """Run the full scoring pipeline for a single ticker.

    Returns (detected_signals, monitor_signals).
        - detected_signals: signals from current detector matches.
        - monitor_signals : synthesized signals from active-signal monitoring
                            (detector returned no match but a prior alert was active).
    """
    log.info("scanning %s", ticker)

    # 1. fetch bars
    daily = fetch_daily_bars(client, ticker, settings.daily_lookback_days)
    if daily.empty or len(daily) < 60:
        log.info("%s: insufficient daily bars (%d)", ticker, len(daily))
        return [], []
    daily = add_common_indicators(daily)

    hourly = _fetch_hourly(client, settings, ticker, asof)
    if len(hourly) < 25:
        log.info("%s: insufficient hourly bars after alignment (%d)", ticker, len(hourly))
        return [], []

    # 2. liquidity prefilter
    last = daily.iloc[-1]
    price = float(last["close"])
    dollar_vol = float(last["close"] * last["volume"])
    if price < settings.min_price or dollar_vol < settings.min_dollar_volume:
        log.info("%s: liquidity prefilter (price=%.2f, $vol=%.0f)", ticker, price, dollar_vol)
        return [], []

    # 3. detect patterns
    enabled = list(settings.enabled_patterns) if settings.enabled_patterns else None
    matches = detect_all(daily, hourly, enabled=enabled)

    # 4. compute common features once (independent of pattern)
    trend_blob = trend_regime(daily, market_ctx.spy_daily)
    comp_blob = volatility_compression(daily)
    sr_blob = sr_quality(daily)
    earn_days = (market_ctx.earnings_days_by_ticker or {}).get(ticker)
    market_blob = market_context(market_ctx.spy_daily, market_ctx.vix_daily, earn_days)
    liq_blob = liquidity(daily)

    detected_signals: list[Signal] = []
    detected_keys: set[str] = set()

    for match in matches:
        # 5. pattern-specific
        geom_blob = score_geometry(match, daily)
        vol_blob = volume_confirmation(daily, hourly, match)
        br_blob = breakout_readiness(daily, hourly, match)
        risk_blob = risk_definition(daily, match)

        common = CommonScores(
            trend=trend_blob.get("_score", 50.0),
            geometry=geom_blob.get("_score", 50.0),
            compression=comp_blob.get("_score", 50.0),
            sr_quality=sr_blob.get("_score", 50.0),
            volume=vol_blob.get("_score", 50.0),
            readiness=br_blob.get("_score", 50.0),
            risk=risk_blob.get("_score", 50.0),
            market=market_blob.get("_score", 50.0),
            liquidity=liq_blob.get("_score", 50.0),
        )

        components = apply_weights(common, geom_blob.get("_score", 50.0), match.pattern)
        raw_score = components.total()

        features_blob: dict[str, Any] = {
            "trend": trend_blob,
            "geometry": geom_blob,
            "compression": comp_blob,
            "sr_quality": sr_blob,
            "volume": vol_blob,
            "readiness": br_blob,
            "risk": risk_blob,
            "market": market_blob,
            "liquidity": liq_blob,
            "pattern_structure": match.structure,
            "pattern_geometry_features": match.geometry_features,
            "_schema_version": "1.0",
        }

        final_score, veto_reasons = apply_vetoes(raw_score, common, match, features_blob)

        # 6. state machine — feed prior_state from the state file (review §P0-1)
        bsb = match.structure.get("bars_since_breakout")
        d_atr = br_blob.get("_distance_atr")
        prior_entry = alert_state.get(ticker, match.pattern)
        prior_state_str = prior_entry.get("state")
        prior_state = None
        try:
            if prior_state_str:
                prior_state = PatternState(prior_state_str)
        except ValueError:
            prior_state = None

        ctx = StateContext(
            final_score=final_score,
            geometry_score=geom_blob.get("_score", 0),
            readiness_score=br_blob.get("_score", 0),
            breakout_distance_atr=d_atr,
            closed_above_trigger=bool(match.neckline and price >= match.neckline),
            closed_below_invalid=bool(match.invalid_below and price <= match.invalid_below),
            bars_since_breakout=bsb,
            volume_confirmed=_is_volume_confirmed(features_blob),
        )
        new_state = determine_state(match, ctx, prior_state=prior_state,
                                     min_candidate=settings.min_candidate_score)

        signal = Signal(
            run_id=run_id,
            asof=asof.isoformat(),
            ticker=ticker,
            pattern=match.pattern,
            pattern_state=new_state.value,
            final_score=final_score,
            common_scores=common,
            components=components,
            veto_triggered=bool(veto_reasons),
            veto_reasons=veto_reasons,
            price=price,
            trigger=match.neckline,
            invalid_below=match.invalid_below,
            measured_move_target=match.measured_move_target,
            features_json=features_blob,
        )
        detected_signals.append(signal)
        detected_keys.add(match.pattern)

    # 7. Save state for ALL detected signals (review §P1) — so prior_state and
    #    the active monitor can rely on a complete picture next cycle.
    for sig in detected_signals:
        update_signal_state(alert_state, sig, settings.active_monitoring_trading_days)

    # 8. Active-signal monitoring (review §P0-2):
    #    For each ticker × pattern that was previously active but didn't match
    #    this scan, check whether price has invalidated or failed the breakout.
    common_proto = CommonScores(
        trend=trend_blob.get("_score", 50.0), geometry=0.0,
        compression=comp_blob.get("_score", 50.0),
        sr_quality=sr_blob.get("_score", 50.0),
        volume=0.0, readiness=0.0, risk=0.0,
        market=market_blob.get("_score", 50.0),
        liquidity=liq_blob.get("_score", 50.0),
    )
    monitor_signals = _monitor_active_for_ticker(
        ticker=ticker, daily=daily, hourly=hourly,
        alert_state=alert_state, detected_keys=detected_keys,
        run_id=run_id, asof=asof, settings=settings,
        common_scores_proto=common_proto,
    )
    return detected_signals, monitor_signals


# ---------------------------------------------------------------------------
# Active signal tracker (review §P0-2)
# ---------------------------------------------------------------------------

def _monitor_active_for_ticker(*, ticker: str, daily: pd.DataFrame, hourly: pd.DataFrame,
                                 alert_state: AlertState, detected_keys: set[str],
                                 run_id: str, asof: datetime, settings: Settings,
                                 common_scores_proto: CommonScores) -> list[Signal]:
    """Generate INVALIDATED / BREAKOUT_FAILED signals for this ticker's active
    prior signals that are no longer detected this cycle."""
    out: list[Signal] = []
    price = float(daily["close"].iloc[-1])

    for entry_ticker, pattern, entry in alert_state.all_entries():
        if entry_ticker.upper() != ticker.upper():
            continue
        if pattern in detected_keys:
            continue
        st_str = entry.get("state")
        if not st_str:
            continue
        try:
            prior_state = PatternState(st_str)
        except ValueError:
            continue
        if prior_state not in ACTIVE_MONITORING_STATES:
            continue

        # Has the monitoring window expired?
        until = entry.get("monitored_until")
        if until:
            try:
                until_ts = pd.Timestamp(until)
                if pd.Timestamp(asof) > until_ts:
                    log.info("active monitor expired: %s::%s", ticker, pattern)
                    alert_state.delete(ticker, pattern)
                    continue
            except Exception:
                pass

        trigger = entry.get("trigger")
        invalid_below = entry.get("invalid_below")
        if trigger is None or invalid_below is None:
            continue

        new_state: PatternState | None = None
        if price <= float(invalid_below):
            new_state = PatternState.INVALIDATED
        elif prior_state == PatternState.BREAKOUT_CONFIRMED and price < float(trigger):
            new_state = PatternState.BREAKOUT_FAILED

        if new_state is None:
            continue

        sig = Signal(
            run_id=run_id, asof=asof.isoformat(), ticker=ticker, pattern=pattern,
            pattern_state=new_state.value, final_score=0.0,
            common_scores=common_scores_proto, components=WeightedComponents(),
            veto_triggered=False, veto_reasons=[],
            price=price, trigger=float(trigger), invalid_below=float(invalid_below),
            measured_move_target=None,
            features_json={"_source": "active_monitor",
                            "_prior_state": st_str,
                            "_schema_version": "1.0"},
            note="active_monitor_event",
        )
        out.append(sig)
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

# Stable column order — the forward-returns tracker relies on it.
_CSV_COLUMNS = [
    "run_id", "asof", "ticker", "pattern", "pattern_state", "final_score",
    "trend", "geometry", "compression", "sr_quality", "volume", "readiness",
    "risk", "market", "liquidity",
    "comp_trend", "comp_geometry", "comp_compression", "comp_sr", "comp_volume",
    "comp_readiness", "comp_risk", "comp_market", "comp_liquidity",
    "veto_triggered", "veto_reasons",
    "price", "trigger", "invalid_below", "measured_move_target",
    "features_json", "note",
    "ret_1h", "ret_eod", "ret_1d", "ret_3d", "ret_5d",
    "max_dd_5d", "max_fav_5d", "touched_invalid_5d", "post_hoc_tag",
]


def write_results(settings: Settings, run_id: str, asof: datetime,
                   signals: list[Signal], errors: list[dict] | None = None,
                   counters: dict | None = None) -> dict[str, Path]:
    out_dir = settings.absolute_result_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = asof.strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}

    snapshot_csv = out_dir / f"scan_{stamp}.csv"
    if signals:
        df = pd.DataFrame([s.to_csv_row() for s in signals]).reindex(columns=_CSV_COLUMNS)
        df = df.sort_values("final_score", ascending=False)
        df.to_csv(snapshot_csv, index=False)
        paths["snapshot_csv"] = snapshot_csv

        log_csv = out_dir / "signals_log.csv"
        if log_csv.exists():
            df.to_csv(log_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(log_csv, index=False)
        paths["signals_log"] = log_csv

    # Review §P1: always rewrite latest_top_signals (even when empty) to prevent stale views.
    latest_top = out_dir / "latest_top_signals.csv"
    if signals:
        df_top = pd.DataFrame([s.to_csv_row() for s in signals]).reindex(columns=_CSV_COLUMNS)
        df_top = df_top.sort_values("final_score", ascending=False).head(50)
    else:
        df_top = pd.DataFrame(columns=_CSV_COLUMNS)
    df_top.to_csv(latest_top, index=False)
    paths["latest_top"] = latest_top

    summary = {
        "run_id": run_id,
        "asof": asof.isoformat(),
        "n_signals": len(signals),
        "n_alerts_eligible": sum(1 for s in signals if s.final_score >= settings.min_alert_score),
        "by_pattern": _count_by(signals, "pattern"),
        "by_state": _count_by(signals, "pattern_state"),
        "counters": counters or {},
        "errors": errors or [],
        "top_5": [
            {"ticker": s.ticker, "pattern": s.pattern, "score": s.final_score, "state": s.pattern_state}
            for s in sorted(signals, key=lambda x: x.final_score, reverse=True)[:5]
        ],
    }
    summary_path = out_dir / f"run_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary"] = summary_path

    latest_summary = out_dir / "latest_run_summary.json"
    latest_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["latest_summary"] = latest_summary
    return paths


def _count_by(signals: list[Signal], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in signals:
        v = getattr(s, attr, "?")
        out[v] = out.get(v, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run() -> int:
    settings = load_settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("scanner v%s starting (dry_run=%s, hourly_align=%s, enabled_patterns=%s)",
             __import__("scanner").__version__, settings.dry_run,
             settings.hourly_alignment_mode,
             ",".join(settings.enabled_patterns) if settings.enabled_patterns else "all")

    if not settings.polygon_api_key:
        log.error("POLYGON_API_KEY not set; aborting")
        return 2

    tickers = load_tickers(settings.absolute_ticker_file())
    log.info("loaded %d tickers from %s", len(tickers), settings.absolute_ticker_file())
    if not tickers:
        log.warning("ticker.txt empty; nothing to scan")
        return 0

    client = PolygonClient(settings)
    asof = datetime.now(timezone.utc)
    run_id = f"{asof.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    state = AlertState(settings)
    market_ctx = fetch_market_context(client, settings)

    all_signals: list[Signal] = []
    n_alerts = 0
    errors: list[dict] = []
    counters: dict = {"detector_hits": {}, "active_monitor_events": 0,
                       "alerts_by_type": {}, "tickers_processed": 0,
                       "tickers_with_errors": 0}

    for ticker in tickers:
        try:
            detected, monitored = scan_ticker(client, settings, ticker, run_id, asof, market_ctx, state)
        except PolygonError as e:
            log.warning("%s: polygon error %s", ticker, e)
            errors.append({"ticker": ticker, "phase": "scan", "error": str(e)[:200]})
            counters["tickers_with_errors"] += 1
            continue
        except Exception as e:  # pragma: no cover
            log.exception("%s: unexpected error %s", ticker, e)
            errors.append({"ticker": ticker, "phase": "scan", "error": str(e)[:200]})
            counters["tickers_with_errors"] += 1
            continue

        counters["tickers_processed"] += 1
        for sig in detected:
            counters["detector_hits"][sig.pattern] = counters["detector_hits"].get(sig.pattern, 0) + 1
        all_signals.extend(detected)
        all_signals.extend(monitored)

        # Active-monitor synthesized signals: each one ALWAYS triggers an alert decision.
        for sig in monitored:
            counters["active_monitor_events"] += 1
            rec = maybe_alert(settings, state, sig)
            if rec:
                n_alerts += 1
                counters["alerts_by_type"][rec.alert_type] = \
                    counters["alerts_by_type"].get(rec.alert_type, 0) + 1
                log.info("active alert: %s %s %s", rec.ticker, rec.pattern, rec.alert_type)

        # Best detected signal earns the regular alert path.
        if detected:
            best = max(detected, key=lambda s: s.final_score)
            if (best.final_score >= settings.min_candidate_score
                    or best.pattern_state in ALERTABLE_STATES):
                rec = maybe_alert(settings, state, best)
                if rec:
                    n_alerts += 1
                    counters["alerts_by_type"][rec.alert_type] = \
                        counters["alerts_by_type"].get(rec.alert_type, 0) + 1
                    log.info("alert sent: %s %s %s (%.1f)",
                             rec.ticker, rec.pattern, rec.alert_type, rec.final_score)

        # Save state mid-loop (review §P2: don't lose progress on timeout)
        try:
            state.save()
        except Exception as e:  # pragma: no cover
            log.warning("intermediate state save failed: %s", e)

    state.save()
    paths = write_results(settings, run_id, asof, all_signals,
                          errors=errors, counters=counters)
    log.info("scan complete: %d signals (incl. %d active-monitor), %d alerts dispatched",
             len(all_signals), counters["active_monitor_events"], n_alerts)
    log.info("artifacts: %s", {k: str(v) for k, v in paths.items()})
    return 0
