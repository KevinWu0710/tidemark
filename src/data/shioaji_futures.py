from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.core.config import ROOT

load_dotenv(ROOT / ".env")

# TMF session slices for RangeTime (same calendar `date`; overnight 00–05 on same date key).
# Ends are exclusive-ish: next slice starts at previous end to limit boundary double-count.
_TMF_HALF_RANGES: tuple[tuple[str, str], ...] = (
    ("08:45:00", "11:15:00"),
    ("11:15:01", "13:45:00"),
    ("15:00:00", "22:00:00"),
    ("22:00:01", "23:59:59"),
    ("00:00:00", "05:00:00"),
)

_TMF_HOUR_RANGES: tuple[tuple[str, str], ...] = (
    ("08:45:00", "09:45:00"),
    ("09:45:01", "10:45:00"),
    ("10:45:01", "11:45:00"),
    ("11:45:01", "12:45:00"),
    ("12:45:01", "13:45:00"),
    ("15:00:00", "16:00:00"),
    ("16:00:01", "17:00:00"),
    ("17:00:01", "18:00:00"),
    ("18:00:01", "19:00:00"),
    ("19:00:01", "20:00:00"),
    ("20:00:01", "21:00:00"),
    ("21:00:01", "22:00:00"),
    ("22:00:01", "23:00:00"),
    ("23:00:01", "23:59:59"),
    ("00:00:00", "01:00:00"),
    ("01:00:01", "02:00:00"),
    ("02:00:01", "03:00:00"),
    ("03:00:01", "04:00:00"),
    ("04:00:01", "05:00:00"),
)


class ShioajiFutures:
    """
    永豐 Shioaji — TMF historical ticks / kbars.

    TMFR1 = 微型台指「近月連續契約」代碼（Rolled front month）。
    歷史回補請優先用 TMFR1；當月代碼（如 TMFK6）在舊日期常回空。

    ticks 預設 AllDay；空則改 RangeTime 半天／小時切片合併（流量未超限時
    仍可能整天皆空＝SIM 無該日 tick）。
    """

    CONTINUOUS_CODE = "TMFR1"
    TICK_CHUNK_PAUSE_SEC = 0.35

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
            logger.info(f"Shioaji login ok sim={self.simulation}")
            return True
        except Exception as e:
            logger.error(f"Shioaji login failed: {e}")
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

    def _pick_tmf_contract(self, date: str, *, prefer_continuous: bool = True):
        assert self.api is not None
        contracts = list(self.api.Contracts.Futures.TMF)
        if prefer_continuous:
            for c in contracts:
                if str(c.code) == self.CONTINUOUS_CODE:
                    return c
        yyyymm = date.replace("-", "")[:6]
        dated = [
            c
            for c in contracts
            if c.delivery_month == yyyymm and not str(c.code).startswith("TMFR")
        ]
        if dated:
            return dated[0]
        future = sorted(
            [
                c
                for c in contracts
                if c.delivery_month >= yyyymm and not str(c.code).startswith("TMFR")
            ],
            key=lambda c: c.delivery_month,
        )
        if future:
            return future[0]
        return contracts[0]

    def _ticks_df(self, raw, contract) -> pd.DataFrame:
        df = pd.DataFrame({**raw})
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], unit="ns")
        df["price"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["contract_code"] = contract.code
        df["contract_date"] = contract.delivery_month
        df["futures_id"] = "TMF"
        df["date"] = df["ts"]
        return df

    def _ticks_range(
        self,
        contract,
        date: str,
        time_start: str,
        time_end: str,
    ) -> pd.DataFrame:
        import shioaji as sj

        assert self.api is not None
        qt = getattr(sj, "TicksQueryType", None) or sj.constant.TicksQueryType
        raw = self.api.ticks(
            contract,
            date=date,
            query_type=qt.RangeTime,
            time_start=time_start,
            time_end=time_end,
        )
        return self._ticks_df(raw, contract)

    def _ticks_chunked(
        self,
        contract,
        date: str,
        *,
        chunk: str = "half",
    ) -> pd.DataFrame:
        ranges = _TMF_HOUR_RANGES if chunk == "hour" else _TMF_HALF_RANGES
        parts: list[pd.DataFrame] = []
        for i, (t0, t1) in enumerate(ranges):
            try:
                part = self._ticks_range(contract, date, t0, t1)
            except Exception as e:
                logger.warning(f"shioaji ticks range {date} {t0}-{t1} failed: {e}")
                part = pd.DataFrame()
            if part is not None and not part.empty:
                parts.append(part)
                logger.info(
                    f"shioaji ticks chunk {contract.code} {date} {t0}-{t1} rows={len(part)}"
                )
            if i + 1 < len(ranges):
                time.sleep(self.TICK_CHUNK_PAUSE_SEC)
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        # 同一 ts 可有多筆成交；只去掉切片邊界造成的「整列完全重複」
        out = out.drop_duplicates(keep="last").sort_values("ts")
        return out.reset_index(drop=True)

    def ticks(
        self,
        date: str,
        contract_code: Optional[str] = None,
        *,
        mode: str = "auto",
        chunk: str = "half",
    ) -> pd.DataFrame:
        """
        mode:
          allday — 一次 AllDay
          chunks — 只走 RangeTime 半天／小時切片
          auto — 先 AllDay；空則改 chunks（預設）
        chunk: half | hour（僅 chunks／auto 後備）
        """
        if not self.connect():
            return pd.DataFrame()
        try:
            if contract_code:
                contract = self.api.Contracts.Futures.TMF[contract_code]
            else:
                contract = self._pick_tmf_contract(date, prefer_continuous=True)

            mode_l = (mode or "auto").lower()
            chunk_l = (chunk or "half").lower()
            if chunk_l not in ("half", "hour"):
                chunk_l = "half"

            df = pd.DataFrame()
            if mode_l in ("allday", "auto"):
                raw = self.api.ticks(contract, date=date)
                df = self._ticks_df(raw, contract)
                if not df.empty:
                    logger.info(f"shioaji ticks {contract.code} {date} rows={len(df)} mode=allday")
                    return df
                if mode_l == "allday":
                    return df
                logger.info(
                    f"shioaji ticks {contract.code} {date} AllDay empty → try RangeTime {chunk_l}"
                )

            if mode_l in ("chunks", "auto"):
                df = self._ticks_chunked(contract, date, chunk=chunk_l)
                if not df.empty:
                    logger.info(
                        f"shioaji ticks {contract.code} {date} rows={len(df)} "
                        f"mode=chunks/{chunk_l}"
                    )
                else:
                    logger.warning(
                        f"shioaji ticks {contract.code} {date} empty after chunks/{chunk_l}"
                    )
            return df
        except Exception as e:
            logger.error(f"shioaji ticks failed {date}: {e}")
            return pd.DataFrame()

    def kbars(
        self,
        date: str,
        contract_code: Optional[str] = None,
        *,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self.connect():
            return pd.DataFrame()
        try:
            if contract_code:
                contract = self.api.Contracts.Futures.TMF[contract_code]
            else:
                contract = self._pick_tmf_contract(date, prefer_continuous=True)
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
            df["contract_code"] = contract.code
            df["contract_date"] = contract.delivery_month
            return df
        except Exception as e:
            logger.error(f"shioaji kbars failed {date}: {e}")
            return pd.DataFrame()
