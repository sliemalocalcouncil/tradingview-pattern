#!/usr/bin/env python3
"""Forward-returns tracker — populates ret_*, max_dd_5d, max_fav_5d,
touched_invalid_5d in signals_log.csv after each holding period elapses.

Per review §P0-4: returns are now computed on the NYSE trading-day calendar
rather than a rough timedelta, and `ret_eod` always pulls the same trading
day's regular-session close.

Horizons:
    ret_1h       — 1 hour after asof (intraday close-to-close)
    ret_eod      — same-trading-day regular-session close
    ret_1d/3d/5d — N trading days after asof (close)

5-day stats (computed once after 5 trading days have elapsed):
    max_dd_5d           — largest drawdown from `price` (negative number)
    max_fav_5d          — largest favorable excursion from `price`
    touched_invalid_5d  — bool: did any low ≤ invalid_below in the window?

This script is meant to run as a separate scheduled GH Actions job so the
main scan stays fast.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scanner.config import load_settings
from scanner.polygon_client import PolygonClient, PolygonError

log = logging.getLogger(__name__)

# Trading-day horizons (number of trading sessions after asof's trading day).
# ret_1h is special — uses hourly bars instead of daily.
HORIZONS_TRADING_DAYS = {
    "ret_eod": 0,
    "ret_1d":  1,
    "ret_3d":  3,
    "ret_5d":  5,
}


# ---------------------------------------------------------------------------
# Trading-day calendar helpers
# ---------------------------------------------------------------------------

def _market_calendar():
    try:
        import pandas_market_calendars as mcal
        return mcal.get_calendar("XNYS")
    except Exception:
        return None


def _trading_close_after(asof: datetime, n_trading_days: int, calendar=None) -> datetime | None:
    """Return UTC timestamp of the regular-session close `n_trading_days` after the
    trading day that contains `asof`. n=0 → same trading day's close."""
    cal = calendar if calendar is not None else _market_calendar()
    if cal is None:
        return None
    # Wide enough window to cover holidays / weekends
    end = asof.date() + timedelta(days=n_trading_days * 2 + 14)
    try:
        sched = cal.schedule(start_date=asof.date(), end_date=end)
    except Exception:
        return None
    if sched.empty:
        return None
    # Find the row whose market_close is on or after asof
    closes = sched["market_close"].tolist()
    same_day_idx = None
    for i, c in enumerate(closes):
        c_ts = pd.Timestamp(c)
        if c_ts >= pd.Timestamp(asof):
            same_day_idx = i
            break
    if same_day_idx is None:
        return None
    target_idx = same_day_idx + n_trading_days
    if target_idx >= len(closes):
        return None
    return pd.Timestamp(closes[target_idx]).tz_convert("UTC").to_pydatetime()


def _is_due(target_close: datetime | None, now: datetime, grace_minutes: int = 30) -> bool:
    """Wait until at least `grace_minutes` after the target close (gives Polygon
    time to publish the final aggregate)."""
    if target_close is None:
        return False
    return now >= target_close + timedelta(minutes=grace_minutes)


def _has_value(v) -> bool:
    return v is not None and not (isinstance(v, float) and pd.isna(v)) and v != ""


# ---------------------------------------------------------------------------
# Polygon fetchers
# ---------------------------------------------------------------------------

def _fetch_close_at_or_after_hourly(client: PolygonClient, ticker: str,
                                      when: datetime) -> float | None:
    end = when + timedelta(days=4)
    df = client.aggregates(
        ticker, multiplier=1, timespan="hour",
        date_from=when.strftime("%Y-%m-%d"),
        date_to=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        return None
    df = df[df.index >= pd.Timestamp(when)]
    if df.empty:
        return None
    return float(df["close"].iloc[0])


def _fetch_eod_close_for_trading_day(client: PolygonClient, ticker: str,
                                       trading_day_close_utc: datetime) -> float | None:
    """Return the regular-session daily close for the trading day that ends at
    `trading_day_close_utc`."""
    day = trading_day_close_utc.date()
    df = client.aggregates(
        ticker, multiplier=1, timespan="day",
        date_from=day.strftime("%Y-%m-%d"),
        date_to=day.strftime("%Y-%m-%d"),
    )
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def _fetch_5d_window(client: PolygonClient, ticker: str,
                      start: datetime, end: datetime) -> pd.DataFrame:
    df = client.aggregates(
        ticker, multiplier=1, timespan="day",
        date_from=start.strftime("%Y-%m-%d"),
        date_to=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        return df
    return df[(df.index >= pd.Timestamp(start.date())) & (df.index <= pd.Timestamp(end.date()))]


# ---------------------------------------------------------------------------
# Update logic
# ---------------------------------------------------------------------------

def _update_returns_for_row(df: pd.DataFrame, i: int, client: PolygonClient,
                              now: datetime, calendar) -> int:
    """Returns the number of cells updated in this row."""
    row = df.loc[i]
    asof = row["asof"]
    if pd.isna(asof):
        return 0
    asof_dt: datetime = asof.to_pydatetime()
    n_updates = 0

    # ret_1h
    if "ret_1h" not in df.columns:
        df["ret_1h"] = None
    if not _has_value(row.get("ret_1h")):
        target = asof_dt + timedelta(hours=1)
        if now >= target + timedelta(minutes=10):
            try:
                cls = _fetch_close_at_or_after_hourly(client, row["ticker"], target)
                if cls and cls > 0 and float(row["price"]) > 0:
                    df.at[i, "ret_1h"] = round((cls - float(row["price"])) / float(row["price"]), 6)
                    n_updates += 1
            except PolygonError as e:
                log.warning("%s ret_1h: %s", row["ticker"], e)

    # ret_eod, ret_1d, ret_3d, ret_5d (trading-day based)
    for col, n_td in HORIZONS_TRADING_DAYS.items():
        if col not in df.columns:
            df[col] = None
        if _has_value(row.get(col)):
            continue
        target_close = _trading_close_after(asof_dt, n_td, calendar=calendar)
        if not _is_due(target_close, now):
            continue
        try:
            cls = _fetch_eod_close_for_trading_day(client, row["ticker"], target_close)
        except PolygonError as e:
            log.warning("%s %s: %s", row["ticker"], col, e)
            continue
        if cls is None or cls <= 0 or float(row["price"]) <= 0:
            continue
        df.at[i, col] = round((cls - float(row["price"])) / float(row["price"]), 6)
        n_updates += 1

    # 5-day window stats — only compute once (when ret_5d becomes available)
    needs_5d_stats = (
        not _has_value(row.get("max_dd_5d")) or
        not _has_value(row.get("max_fav_5d")) or
        not _has_value(row.get("touched_invalid_5d"))
    )
    if needs_5d_stats:
        target_close_5d = _trading_close_after(asof_dt, 5, calendar=calendar)
        if _is_due(target_close_5d, now):
            try:
                start = asof_dt
                end = target_close_5d
                window = _fetch_5d_window(client, row["ticker"], start, end)
            except PolygonError as e:
                log.warning("%s 5d window: %s", row["ticker"], e)
                window = pd.DataFrame()
            if not window.empty and float(row["price"]) > 0:
                p = float(row["price"])
                if "max_dd_5d" not in df.columns:
                    df["max_dd_5d"] = None
                if "max_fav_5d" not in df.columns:
                    df["max_fav_5d"] = None
                if "touched_invalid_5d" not in df.columns:
                    df["touched_invalid_5d"] = None
                df.at[i, "max_dd_5d"] = round((float(window["low"].min()) - p) / p, 6)
                df.at[i, "max_fav_5d"] = round((float(window["high"].max()) - p) / p, 6)
                inv = row.get("invalid_below")
                if _has_value(inv):
                    touched = bool((window["low"] <= float(inv)).any())
                    df.at[i, "touched_invalid_5d"] = touched
                n_updates += 3
    return n_updates


def update_returns(signals_log: Path) -> int:
    settings = load_settings()
    if not settings.polygon_api_key:
        log.error("POLYGON_API_KEY not set")
        return 2
    if not signals_log.exists():
        log.info("no signals log yet at %s", signals_log)
        return 0

    df = pd.read_csv(signals_log)
    if "asof" not in df.columns:
        log.error("signals_log.csv missing 'asof' column")
        return 1
    df["asof"] = pd.to_datetime(df["asof"], utc=True, errors="coerce")
    now = datetime.now(timezone.utc)
    client = PolygonClient(settings)
    calendar = _market_calendar()

    total_updates = 0
    for i in df.index:
        try:
            total_updates += _update_returns_for_row(df, i, client, now, calendar)
        except Exception as e:  # pragma: no cover
            log.warning("row %d update failed: %s", i, e)
            continue

    if total_updates > 0:
        df.to_csv(signals_log, index=False)
        log.info("updated %d cells in %s", total_updates, signals_log)
    else:
        log.info("nothing to update")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = load_settings()
    sys.exit(update_returns(settings.absolute_result_dir() / "signals_log.csv"))
