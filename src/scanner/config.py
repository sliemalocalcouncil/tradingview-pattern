"""Configuration — loaded from environment variables with sensible defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0")
    return str(raw).lower() in {"1", "true", "yes", "y", "on"}


def _parse_pattern_list(raw: str | None) -> tuple[str, ...]:
    """ENABLED_PATTERNS env: 'all' or '' = use every detector; else CSV list."""
    if not raw or not raw.strip() or raw.strip().lower() == "all":
        return ()
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


@dataclass(frozen=True)
class Settings:
    # --- API ---
    polygon_api_key: str = field(default_factory=lambda: _env("POLYGON_API_KEY", ""))
    polygon_base_url: str = field(default_factory=lambda: _env("POLYGON_BASE_URL", "https://api.polygon.io"))
    # Default 0 = unlimited (Starter+ tier). Set to 5 for Free tier.
    polygon_req_per_min: int = field(default_factory=lambda: _env_int("POLYGON_REQ_PER_MIN", 0))
    polygon_request_timeout_s: int = field(default_factory=lambda: _env_int("POLYGON_REQUEST_TIMEOUT_S", 30))

    # --- Telegram ---
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID", ""))
    telegram_disabled: bool = field(default_factory=lambda: _env_bool("TELEGRAM_DISABLED", False))

    # --- Universe / IO ---
    project_root: Path = field(default_factory=lambda: Path(_env("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))))
    ticker_file: Path = field(default_factory=lambda: Path(_env("TICKER_FILE", "ticker.txt")))
    result_dir: Path = field(default_factory=lambda: Path(_env("RESULT_DIR", "result")))

    # --- Data windows ---
    # Starter has 5 years of historical data; ~500 daily bars (~2 years) is the
    # sweet spot for VCP / Cup&Handle base detection without slowing the run.
    daily_lookback_days: int = field(default_factory=lambda: _env_int("DAILY_LOOKBACK_DAYS", 500))
    intraday_lookback_days: int = field(default_factory=lambda: _env_int("INTRADAY_LOOKBACK_DAYS", 45))

    # --- Scoring thresholds (v1 starting points; calibrate empirically per design doc §9) ---
    min_candidate_score: float = field(default_factory=lambda: _env_float("MIN_CANDIDATE_SCORE", 60.0))
    min_alert_score: float = field(default_factory=lambda: _env_float("MIN_ALERT_SCORE", 70.0))
    score_upgrade_delta: float = field(default_factory=lambda: _env_float("SCORE_UPGRADE_DELTA", 8.0))
    cooldown_hours: int = field(default_factory=lambda: _env_int("COOLDOWN_HOURS", 6))

    # --- Liquidity guards ---
    # Starter users typically run a larger universe of liquid names; raising
    # min_dollar_volume avoids spending API calls on illiquid tickers.
    min_price: float = field(default_factory=lambda: _env_float("MIN_PRICE", 5.0))
    min_dollar_volume: float = field(default_factory=lambda: _env_float("MIN_DOLLAR_VOLUME", 10_000_000))

    # --- Pattern toggle (noise isolation per review §P1-c) ---
    enabled_patterns: tuple[str, ...] = field(
        default_factory=lambda: _parse_pattern_list(_env("ENABLED_PATTERNS", ""))
    )

    # --- Active signal monitoring (review §P0-2) ---
    active_monitoring_trading_days: int = field(
        default_factory=lambda: _env_int("ACTIVE_MONITORING_TRADING_DAYS", 5)
    )

    # --- Hourly bar alignment (review §P0-3) ---
    # "et_aligned" = resample 30-min bars to ET-anchored 09:30/10:30/11:30/...
    # "native"     = use Polygon native 1-hour bars (legacy, has 30-min skew bug)
    hourly_alignment_mode: str = field(
        default_factory=lambda: _env("HOURLY_ALIGNMENT_MODE", "et_aligned")
    )

    # --- Behavior ---
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", False))
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))

    def absolute_ticker_file(self) -> Path:
        p = self.ticker_file
        return p if p.is_absolute() else self.project_root / p

    def absolute_result_dir(self) -> Path:
        p = self.result_dir
        return p if p.is_absolute() else self.project_root / p


def load_settings() -> Settings:
    s = Settings()
    s.absolute_result_dir().mkdir(parents=True, exist_ok=True)
    return s
