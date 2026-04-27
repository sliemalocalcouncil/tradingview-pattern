"""Bar fetching + 60-minute resampling with session-boundary bug fix.

This addresses two bugs:
  C5 (original): the previous 60-min implementation produced a half-bar at 15:30 ET
      that mixed incomplete and complete bars, biasing volume confirmation downward.
  Review §P0-3: Polygon native 1-hour bars are clock-aligned (e.g., 14:00, 15:00 UTC),
      not session-aligned. During EDT, the 13:00 UTC bar (= 9:00 ET) overlaps the
      market open at 9:30 ET — so the first 30 minutes of the trading day was
      either dropped (legacy filter) or polluted by pre-market.

Fix: fetch 30-minute bars and resample to ET-anchored 60-min bars
(09:30/10:30/11:30/.../15:30 ET starts). This matches what a session-anchored
hourly chart looks like and gives the detectors clean data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import pandas_market_calendars as mcal
    _NYSE = mcal.get_calendar("XNYS")
except Exception:  # pragma: no cover
    mcal = None
    _NYSE = None

from .polygon_client import PolygonClient

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Regular session: 09:30 ET → 16:00 ET = 6.5 hours, so 13 30-min bars per day.
# We pair every two 30-min bars into one 60-min bar, ET-anchored.
SESSION_START_ET_MINUTES = 9 * 60 + 30      # 09:30 ET
SESSION_END_ET_MINUTES   = 16 * 60          # 16:00 ET (regular close)
EARLY_CLOSE_ET_MINUTES   = 13 * 60          # 13:00 ET (NYSE half-day)


def fetch_daily_bars(client: PolygonClient, ticker: str, days_back: int) -> pd.DataFrame:
    """Fetch daily bars; return DataFrame indexed by tz-aware UTC timestamp."""
    date_from, date_to = client.date_window(days_back)
    df = client.aggregates(ticker, multiplier=1, timespan="day",
                           date_from=date_from, date_to=date_to)
    if df.empty:
        return df
    df.attrs["ticker"] = ticker
    df.attrs["timeframe"] = "1d"
    return df


def fetch_intraday_bars(client: PolygonClient, ticker: str, days_back: int,
                        timespan: str = "minute", multiplier: int = 30) -> pd.DataFrame:
    """Fetch intraday bars from Polygon. Defaults to 30-minute bars (used by
    `to_session_aligned_hourly` to build clean ET-anchored 60-min bars)."""
    date_from, date_to = client.date_window(days_back)
    df = client.aggregates(ticker, multiplier=multiplier, timespan=timespan,
                           date_from=date_from, date_to=date_to)
    if df.empty:
        return df
    df.attrs["ticker"] = ticker
    df.attrs["timeframe"] = f"{multiplier}{timespan[0]}"
    return df


def to_session_aligned_hourly(intraday: pd.DataFrame, asof: datetime | None = None) -> pd.DataFrame:
    """Resample 30-min bars to ET-anchored 60-min bars.

    Pairs bars starting at (09:30, 10:00) → 09:30 hourly bar, (10:30, 11:00) → 10:30,
    and so on through (15:30) which is a single half-bar marked incomplete unless
    16:00 has already passed.

    Returns a DataFrame indexed by the bar's start (UTC, tz-aware) with columns:
    open, high, low, close, volume, vwap (if present), complete (bool).
    """
    if intraday.empty:
        return intraday.assign(complete=pd.Series(dtype=bool))
    if asof is None:
        asof = datetime.now(timezone.utc)

    df = intraday.copy()
    et_starts = df.index.tz_convert(ET)
    df["_et_date"] = et_starts.date
    df["_et_min"]  = et_starts.hour * 60 + et_starts.minute

    # Keep only regular-session bars (09:30 .. 16:00 inclusive of starts up to 15:30)
    mask = (df["_et_min"] >= SESSION_START_ET_MINUTES) & (df["_et_min"] < SESSION_END_ET_MINUTES)
    df = df[mask].copy()
    if df.empty:
        return df.drop(columns=["_et_date", "_et_min"]).assign(complete=pd.Series(dtype=bool))

    # Each 60-min bin is keyed by ET date + hour-bucket-start (in 30-min steps from 09:30)
    # bin index 0 = 09:30, 1 = 10:30, 2 = 11:30, ..., 6 = 15:30
    df["_bin"] = ((df["_et_min"] - SESSION_START_ET_MINUTES) // 60).astype(int)
    grp_keys = df.groupby(["_et_date", "_bin"], sort=True)

    rows: list[dict] = []
    has_vwap = "vwap" in df.columns
    for (et_date, bin_idx), grp in grp_keys:
        # Bar start (ET): 09:30 + bin_idx hours
        bin_start_et_min = SESSION_START_ET_MINUTES + bin_idx * 60
        bin_start = pd.Timestamp(et_date, tz=ET) + pd.Timedelta(minutes=bin_start_et_min)
        bin_end_et_min = bin_start_et_min + 60
        out_row = {
            "timestamp": bin_start.tz_convert(timezone.utc),
            "open":   float(grp["open"].iloc[0]),
            "high":   float(grp["high"].max()),
            "low":    float(grp["low"].min()),
            "close":  float(grp["close"].iloc[-1]),
            "volume": float(grp["volume"].sum()),
        }
        if has_vwap and grp["vwap"].notna().any():
            # volume-weighted vwap of components
            v = grp["volume"].fillna(0).values
            w = grp["vwap"].fillna(grp["close"]).values
            out_row["vwap"] = float((v * w).sum() / max(v.sum(), 1e-9))

        # Bar is complete only if (a) two 30-min components present
        # AND (b) bin_end already passed (asof) AND (c) bin_end ≤ market close that day.
        full_pair = len(grp) >= 2
        bin_end = bin_start + pd.Timedelta(minutes=60)
        passed_now = bin_end <= pd.Timestamp(asof)

        market_close_et_min = SESSION_END_ET_MINUTES
        if _NYSE is not None:
            try:
                schedule = _NYSE.schedule(start_date=et_date, end_date=et_date)
                if not schedule.empty:
                    close_utc = schedule.iloc[0]["market_close"]
                    close_et = close_utc.tz_convert(ET)
                    market_close_et_min = close_et.hour * 60 + close_et.minute
            except Exception as e:  # pragma: no cover
                log.debug("market calendar lookup failed: %s", e)
        within_session = bin_end_et_min <= market_close_et_min

        out_row["complete"] = bool(full_pair and passed_now and within_session)
        rows.append(out_row)

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "complete"])
    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return out


def fetch_session_aligned_hourly(client: PolygonClient, ticker: str, days_back: int,
                                   asof: datetime | None = None) -> pd.DataFrame:
    """Convenience: fetch 30-min bars and resample to ET-anchored 60-min bars."""
    raw = fetch_intraday_bars(client, ticker, days_back, timespan="minute", multiplier=30)
    if raw.empty:
        return raw
    return to_session_aligned_hourly(raw, asof=asof)


# ---------------------------------------------------------------------------
# Legacy native-hourly path (kept behind HOURLY_ALIGNMENT_MODE=native flag)
# ---------------------------------------------------------------------------

def fetch_native_hourly(client: PolygonClient, ticker: str, days_back: int) -> pd.DataFrame:
    return fetch_intraday_bars(client, ticker, days_back, timespan="hour", multiplier=1)


def to_regular_session_hourly(intraday: pd.DataFrame, asof: datetime | None = None) -> pd.DataFrame:
    """Legacy path — filter native hourly bars to regular session and tag completeness.
    Has the 30-min skew bug noted above; kept for fallback only."""
    if intraday.empty:
        return intraday.assign(complete=pd.Series(dtype=bool))
    if asof is None:
        asof = datetime.now(timezone.utc)

    df = intraday.copy()
    et_starts = df.index.tz_convert(ET)
    minutes_from_midnight = et_starts.hour * 60 + et_starts.minute
    in_session = (minutes_from_midnight >= SESSION_START_ET_MINUTES) & (minutes_from_midnight <= 900)
    df = df[in_session].copy()
    df["complete"] = True
    bar_end = df.index + pd.Timedelta(hours=1)
    df.loc[bar_end > pd.Timestamp(asof), "complete"] = False
    if _NYSE is not None:
        try:
            start = df.index.min().date()
            end = df.index.max().date()
            schedule = _NYSE.schedule(start_date=start, end_date=end)
            for ts, _row in df.iterrows():
                day = ts.tz_convert(ET).date()
                if day in schedule.index.date:
                    close_utc = schedule.loc[schedule.index.date == day, "market_close"].iloc[0]
                    if (ts + pd.Timedelta(hours=1)) > close_utc:
                        df.at[ts, "complete"] = False
        except Exception as e:  # pragma: no cover
            log.debug("market calendar lookup failed: %s", e)
    return df


def latest_complete_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Drop the trailing incomplete bar(s) so pattern detectors only see closed bars."""
    if hourly.empty or "complete" not in hourly.columns:
        return hourly
    return hourly[hourly["complete"]].drop(columns=["complete"])
