"""Polygon.io REST client.

Implements just the endpoints we need:
  - Aggregates (bars): /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
  - Previous-day snapshot: /v2/aggs/ticker/{ticker}/prev

Designed for both free tier (5 req/min throttling) and paid tiers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from .config import Settings

log = logging.getLogger(__name__)


class PolygonError(Exception):
    pass


@dataclass
class _RateLimiter:
    requests_per_minute: int

    def __post_init__(self) -> None:
        self._timestamps: list[float] = []

    def wait(self) -> None:
        if self.requests_per_minute <= 0:  # 0 = no throttling
            return
        now = time.time()
        # purge older than 60s
        self._timestamps = [t for t in self._timestamps if now - t < 60.0]
        if len(self._timestamps) >= self.requests_per_minute:
            oldest = self._timestamps[0]
            sleep_for = 60.0 - (now - oldest) + 0.05
            if sleep_for > 0:
                log.info("polygon rate limit reached, sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
        self._timestamps.append(time.time())


class PolygonClient:
    def __init__(self, settings: Settings):
        if not settings.polygon_api_key:
            raise PolygonError("POLYGON_API_KEY not set")
        self.settings = settings
        self._session = requests.Session()
        self._limiter = _RateLimiter(requests_per_minute=settings.polygon_req_per_min)

    # -------------------------- low-level --------------------------

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.settings.polygon_base_url}{path}"
        params = dict(params or {})
        params["apiKey"] = self.settings.polygon_api_key

        last_err: Exception | None = None
        for attempt in range(4):
            self._limiter.wait()
            try:
                resp = self._session.get(url, params=params, timeout=self.settings.polygon_request_timeout_s)
            except requests.RequestException as e:
                last_err = e
                log.warning("polygon network error attempt=%d %s", attempt + 1, e)
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "5"))
                log.info("polygon 429, retry after %.1fs", retry_after)
                time.sleep(retry_after)
                continue
            if resp.status_code >= 500:
                last_err = PolygonError(f"server {resp.status_code}: {resp.text[:200]}")
                log.warning("polygon %s, retrying", resp.status_code)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                # 4xx other than 429 are usually permanent (bad ticker, no data)
                raise PolygonError(f"polygon {resp.status_code} on {path}: {resp.text[:200]}")

            try:
                return resp.json()
            except ValueError as e:
                raise PolygonError(f"invalid json from polygon: {e}") from e

        raise PolygonError(f"polygon failed after retries on {path}: {last_err}")

    # -------------------------- public --------------------------

    def aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,  # "minute" | "hour" | "day"
        date_from: str,  # YYYY-MM-DD
        date_to: str,    # YYYY-MM-DD
        adjusted: bool = True,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """Return OHLCV bars as a DataFrame indexed by UTC timestamp.

        Returned columns: open, high, low, close, volume, vwap (optional), txn (optional).
        """
        path = f"/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{date_from}/{date_to}"
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit,
        }
        data = self._request(path, params=params)
        results = data.get("results") or []
        if not results:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(results)
        # Polygon column rename: t=ms epoch, o, h, l, c, v, vw, n
        rename = {"t": "ts_ms", "o": "open", "h": "high", "l": "low",
                  "c": "close", "v": "volume", "vw": "vwap", "n": "txn"}
        df = df.rename(columns=rename)
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.set_index("timestamp").drop(columns=["ts_ms"], errors="ignore")
        keep = [c for c in ["open", "high", "low", "close", "volume", "vwap", "txn"] if c in df.columns]
        return df[keep].sort_index()

    def previous_day(self, ticker: str) -> dict[str, Any] | None:
        path = f"/v2/aggs/ticker/{ticker.upper()}/prev"
        data = self._request(path, params={"adjusted": "true"})
        results = data.get("results") or []
        if not results:
            return None
        r = results[0]
        return {
            "open":   r.get("o"),
            "high":   r.get("h"),
            "low":    r.get("l"),
            "close":  r.get("c"),
            "volume": r.get("v"),
            "vwap":   r.get("vw"),
        }

    @staticmethod
    def date_window(days_back: int, end: datetime | None = None) -> tuple[str, str]:
        if end is None:
            end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
