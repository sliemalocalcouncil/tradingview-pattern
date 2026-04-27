"""Data classes for pattern matches, signals, alerts, and explanations."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PatternMatch:
    """Geometric pattern detection — knows nothing about scoring."""
    pattern: str
    direction: str = "long"
    neckline: float | None = None       # primary breakout/trigger level
    invalid_below: float | None = None
    measured_move_target: float | None = None
    structure: dict[str, Any] = field(default_factory=dict)  # pattern-specific landmarks
    geometry_features: dict[str, float] = field(default_factory=dict)  # feature → 0..1 or raw
    notes: str = ""


@dataclass
class CommonScores:
    trend: float
    geometry: float          # set after geometry scoring
    compression: float
    sr_quality: float
    volume: float
    readiness: float
    risk: float
    market: float
    liquidity: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class WeightedComponents:
    """final = sum of components; each component = score × pattern weight."""
    trend: float = 0.0
    geometry: float = 0.0
    compression: float = 0.0
    sr_quality: float = 0.0
    volume: float = 0.0
    readiness: float = 0.0
    risk: float = 0.0
    market: float = 0.0
    liquidity: float = 0.0

    def total(self) -> float:
        return sum(asdict(self).values())


@dataclass
class Signal:
    run_id: str
    asof: str  # ISO timestamp UTC
    ticker: str
    pattern: str
    pattern_state: str  # state machine value
    final_score: float
    common_scores: CommonScores
    components: WeightedComponents
    veto_triggered: bool
    veto_reasons: list[str]
    price: float
    trigger: float | None
    invalid_below: float | None
    measured_move_target: float | None
    features_json: dict[str, Any]  # full breakdown
    note: str = ""

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "asof": self.asof,
            "ticker": self.ticker,
            "pattern": self.pattern,
            "pattern_state": self.pattern_state,
            "final_score": round(self.final_score, 2),
            "trend": round(self.common_scores.trend, 1),
            "geometry": round(self.common_scores.geometry, 1),
            "compression": round(self.common_scores.compression, 1),
            "sr_quality": round(self.common_scores.sr_quality, 1),
            "volume": round(self.common_scores.volume, 1),
            "readiness": round(self.common_scores.readiness, 1),
            "risk": round(self.common_scores.risk, 1),
            "market": round(self.common_scores.market, 1),
            "liquidity": round(self.common_scores.liquidity, 1),
            "comp_trend": round(self.components.trend, 2),
            "comp_geometry": round(self.components.geometry, 2),
            "comp_compression": round(self.components.compression, 2),
            "comp_sr": round(self.components.sr_quality, 2),
            "comp_volume": round(self.components.volume, 2),
            "comp_readiness": round(self.components.readiness, 2),
            "comp_risk": round(self.components.risk, 2),
            "comp_market": round(self.components.market, 2),
            "comp_liquidity": round(self.components.liquidity, 2),
            "veto_triggered": self.veto_triggered,
            "veto_reasons": ";".join(self.veto_reasons),
            "price": round(self.price, 4),
            "trigger": round(self.trigger, 4) if self.trigger else None,
            "invalid_below": round(self.invalid_below, 4) if self.invalid_below else None,
            "measured_move_target": round(self.measured_move_target, 4) if self.measured_move_target else None,
            "features_json": json.dumps(self.features_json, default=str, separators=(",", ":")),
            "note": self.note,
            # forward returns — populated by the forward returns tracker job
            "ret_1h": None,
            "ret_eod": None,
            "ret_1d": None,
            "ret_3d": None,
            "ret_5d": None,
            "max_dd_5d": None,
            "max_fav_5d": None,
            "touched_invalid_5d": None,
            "post_hoc_tag": None,
        }


@dataclass
class AlertRecord:
    run_id: str
    asof: str
    ticker: str
    pattern: str
    pattern_state: str
    alert_type: str
    final_score: float
    price: float
    trigger: float | None
    invalid_below: float | None
    note: str = ""
