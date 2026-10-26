from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Side(str, Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class Session(str, Enum):
    DAY = "day"
    NIGHT = "night"


class TradeAction(str, Enum):
    HOLD = "HOLD"
    BUY = "BUY"       # open / add long (本階段最多 1 口)
    SELL = "SELL"     # open / add short
    CLOSE = "CLOSE"   # flatten


@dataclass(frozen=True)
class SignalDecision:
    """Pure signal output — no I/O, no broker calls."""

    action: TradeAction
    reason: str
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    ts: Optional[str] = None  # ISO timestamp when evaluated
