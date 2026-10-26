"""
即時 tick → 1 分鐘 OHLCV（因果：分鐘結束才關閉那根 K）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.data.sessions import label_session


def _floor_minute(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.floor("min")


@dataclass
class MinuteBarBuilder:
    """
    餵入逐筆 tick；當分鐘換檔時回傳「剛結束」的完整 1m bar。
    未結束的當根可經 current_bar() 窺看（策略預設只吃已關閉棒）。
    """

    _bucket: Optional[pd.Timestamp] = None
    _open: float = 0.0
    _high: float = 0.0
    _low: float = 0.0
    _close: float = 0.0
    _volume: float = 0.0
    _n: int = 0

    def on_tick(
        self,
        ts: Any,
        price: float,
        volume: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        t = pd.Timestamp(ts)
        px = float(price)
        vol = float(volume or 0.0)
        if pd.isna(t) or px <= 0:
            return None
        bucket = _floor_minute(t)
        closed: Optional[dict[str, Any]] = None
        if self._bucket is None:
            self._start(bucket, px, vol)
            return None
        if bucket != self._bucket:
            closed = self._as_bar(self._bucket)
            self._start(bucket, px, vol)
            return closed
        self._high = max(self._high, px)
        self._low = min(self._low, px)
        self._close = px
        self._volume += vol
        self._n += 1
        return None

    def flush(self) -> Optional[dict[str, Any]]:
        """盤終或停止訂閱時關閉最後一根未完成棒。"""
        if self._bucket is None or self._n <= 0:
            return None
        bar = self._as_bar(self._bucket)
        self._bucket = None
        self._n = 0
        return bar

    def current_bar(self) -> Optional[dict[str, Any]]:
        if self._bucket is None or self._n <= 0:
            return None
        return self._as_bar(self._bucket)

    def _start(self, bucket: pd.Timestamp, px: float, vol: float) -> None:
        self._bucket = bucket
        self._open = self._high = self._low = self._close = px
        self._volume = vol
        self._n = 1

    def _as_bar(self, bucket: pd.Timestamp) -> dict[str, Any]:
        sess = label_session(bucket)
        return {
            "ts": bucket,
            "open": float(self._open),
            "high": float(self._high),
            "low": float(self._low),
            "close": float(self._close),
            "volume": float(self._volume),
            "session": sess,
        }


def bars_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume", "session"]
        )
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)
