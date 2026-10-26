"""共用意圖型別（無 backtest／券商依賴，避免循環 import）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.core.types import TradeAction


@dataclass(frozen=True)
class TradeIntent:
    """訊號意圖（尚未過策略風控／下單）。"""

    ts: Any
    action: TradeAction
    price: float
    reason: str
    track: Optional[str] = None

    def key(self) -> str:
        return f"{self.ts}|{self.action.value}|{self.reason}"
