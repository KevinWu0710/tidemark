from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Tuple, Union

import pandas as pd

TsLike = Union[pd.Timestamp, datetime, str]


def _as_ts(ts: TsLike) -> pd.Timestamp:
    if isinstance(ts, str):
        return pd.Timestamp(ts)
    if isinstance(ts, pd.Timestamp):
        return ts
    return pd.Timestamp(ts)


def label_session(ts: TsLike) -> str:
    """
    Label clock time as day / night for TMF.

    FinMind tick file for calendar date D (see docs/ai/DATA.md):
    - 00:00–05:00 → night (after_market attributed to D)
    - 08:45–13:45 → day
    - 15:00–24:00 → night (after_market attributed to next trading day)
    """
    t = _as_ts(ts).time()
    if time(8, 45) <= t <= time(13, 45):
        return "day"
    if t >= time(15, 0) or t <= time(5, 0):
        return "night"
    return "closed"


def trading_day_for(ts: TsLike) -> str:
    """
    交易日標籤（字串 YYYY-MM-DD），日／夜分開統計用。

    - day 08:45–13:45 → 當日
    - night 00:00–05:00 → 當日（屬該曆日 D 的夜盤尾段）
    - night 15:00–24:00 → 次曆日（屬下一交易日的夜盤）
    - closed → 當日（僅後備）
    """
    t = _as_ts(ts)
    sess = label_session(t)
    d = t.date()
    if sess == "night" and t.time() >= time(15, 0):
        d = d + timedelta(days=1)
    return str(d)


def session_and_trading_day(ts: TsLike) -> Tuple[str, str]:
    return label_session(ts), trading_day_for(ts)


def is_tradable_session(session: str) -> bool:
    return session in ("day", "night")
