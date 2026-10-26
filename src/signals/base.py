from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.core.types import Side, SignalDecision


class BaseSignal(ABC):
    """Signals are pure: given history ending at t, return a decision."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def evaluate(
        self,
        history: pd.DataFrame,
        position: Side,
        *,
        now_ts: Any = None,
    ) -> SignalDecision:
        raise NotImplementedError
