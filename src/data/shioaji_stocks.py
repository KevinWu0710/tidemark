from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.core.config import ROOT

load_dotenv(ROOT / ".env")


class ShioajiStocks:
    """永豐 Shioaji — 現股歷史 kbars（1m）／ticks。"""

    def __init__(self, simulation: bool = True):
        self.simulation = simulation
        self.api = None
        if simulation:
            self.api_key = os.environ.get("SHIOAJI_SIM_API_KEY", "")
            self.secret_key = os.environ.get("SHIOAJI_SIM_SECRET_KEY", "")
        else:
            self.api_key = os.environ.get("SHIOAJI_LIVE_API_KEY", "")
            self.secret_key = os.environ.get("SHIOAJI_LIVE_SECRET_KEY", "")

    def connect(self) -> bool:
        if self.api is not None:
            return True
        if not self.api_key or not self.secret_key:
            logger.error("Shioaji API keys missing in .env")
            return False
        try:
            import shioaji as sj

            self.api = sj.Shioaji(simulation=self.simulation)
            self.api.login(api_key=self.api_key, secret_key=self.secret_key)
            logger.info(f"Shioaji stocks login ok sim={self.simulation}")
            return True
        except Exception as e:
            logger.error(f"Shioaji stocks login failed: {e}")
            self.api = None
            return False

    def close(self) -> None:
        if self.api is None:
            return
        try:
            self.api.logout()
        except Exception:
            pass
        self.api = None

    def _contract(self, symbol: str):
        assert self.api is not None
        code = str(symbol).upper().replace(".TW", "")
        return self.api.Contracts.Stocks[code]

    def kbars(self, symbol: str, date: str, *, end_date: Optional[str] = None) -> pd.DataFrame:
        if not self.connect():
            return pd.DataFrame()
        try:
            contract = self._contract(symbol)
            end = end_date or date
            raw = self.api.kbars(contract, start=date, end=end)
            df = pd.DataFrame({**raw})
            if df.empty:
                return df
            df["ts"] = pd.to_datetime(df["ts"], unit="ns")
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df["symbol"] = str(symbol).upper()
            return df
        except Exception as e:
            logger.error(f"shioaji stock kbars failed {symbol} {date}: {e}")
            return pd.DataFrame()

    def ticks(self, symbol: str, date: str) -> pd.DataFrame:
        if not self.connect():
            return pd.DataFrame()
        try:
            contract = self._contract(symbol)
            raw = self.api.ticks(contract, date=date)
            df = pd.DataFrame({**raw})
            if df.empty:
                return df
            df["ts"] = pd.to_datetime(df["ts"], unit="ns")
            df["price"] = pd.to_numeric(df["close"], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            df["symbol"] = str(symbol).upper()
            df["date"] = df["ts"]
            return df
        except Exception as e:
            logger.error(f"shioaji stock ticks failed {symbol} {date}: {e}")
            return pd.DataFrame()
