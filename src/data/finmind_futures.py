from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

from src.core.config import ROOT

load_dotenv(ROOT / ".env")

API_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindFutures:
    """Minimal FinMind client for Taiwan futures tick / daily (Sponsor)."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("FINMIND_TOKEN", "")
        if not self.token:
            logger.warning("FINMIND_TOKEN missing — fetch will fail")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _get(self, params: dict, retries: int = 3) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.get(
                    API_URL,
                    headers=self._headers(),
                    params=params,
                    timeout=(10, 120),
                )
                if resp.status_code == 402:
                    logger.error("FinMind 402 api_request_limit — back off")
                    time.sleep(2 ** attempt)
                    continue
                payload = resp.json()
                status = payload.get("status")
                if status not in (200, "success", None) and payload.get("msg"):
                    # FinMind often uses status 200 int
                    if status != 200:
                        logger.warning(f"FinMind msg: {payload.get('msg')}")
                data = payload.get("data") or []
                if not data:
                    return pd.DataFrame()
                return pd.DataFrame(data)
            except Exception as e:
                last_err = e
                logger.warning(f"FinMind request failed ({attempt+1}/{retries}): {e}")
                time.sleep(1 + attempt)
        if last_err:
            raise last_err
        return pd.DataFrame()

    def futures_tick(self, futures_id: str, date: str) -> pd.DataFrame:
        """
        TaiwanFuturesTick — one calendar day per call; data_id required.
        Example: futures_id='TMF', date='2026-03-20'
        """
        df = self._get(
            {
                "dataset": "TaiwanFuturesTick",
                "data_id": futures_id,
                "start_date": date,
            }
        )
        return df

    def futures_daily(
        self, futures_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        return self._get(
            {
                "dataset": "TaiwanFuturesDaily",
                "data_id": futures_id,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

    def futures_kbar(self, futures_id: str, date: str) -> pd.DataFrame:
        """TaiwanFuturesKBar — minute bars, one day."""
        return self._get(
            {
                "dataset": "TaiwanFuturesKBar",
                "data_id": futures_id,
                "start_date": date,
            }
        )
