from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.core.config import ROOT, load_settings

# 各 kind 只留回測需要的欄 → 檔更小、讀更快
_KEEP_COLS = {
    "tick": ["ts", "price", "volume", "session", "contract_code", "contract_date"],
    "1s": ["ts", "open", "high", "low", "close", "volume", "session", "contract_date"],
    "1m": ["ts", "open", "high", "low", "close", "volume", "session", "contract_code", "contract_date"],
}


class LocalStore:
    """
    Persist market data for offline backtests.

    Layout:
      data/cache/{SYMBOL}/{kind}/{YYYY-MM-DD}.parquet

    Legacy readable:
      data/cache/tmf/TMF/...
    """

    def __init__(self, settings: Optional[dict] = None):
        self.settings = settings or load_settings()
        data_cfg = self.settings.get("data", {})
        configured = data_cfg.get("cache_dir", "data/cache")
        self.cache_dir = ROOT / configured
        self._legacy_tmf_root = ROOT / "data" / "cache" / "tmf"
        self.raw_dir = ROOT / data_cfg.get("raw_dir", "data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        # zstd: 小檔；若環境無 zstd codec 則退回 snappy
        self.compression = data_cfg.get("parquet_compression", "zstd")

    def path_for(self, symbol: str, date: str, kind: str = "1s") -> Path:
        return self.cache_dir / symbol.upper() / kind / f"{date}.parquet"

    def _legacy_path(self, symbol: str, date: str, kind: str) -> Path:
        return self._legacy_tmf_root / symbol.upper() / kind / f"{date}.parquet"

    def _slim(self, df: pd.DataFrame, kind: str) -> pd.DataFrame:
        keep = _KEEP_COLS.get(kind)
        if not keep or df is None or df.empty:
            return df
        # normalize tick close→price already done upstream; allow close alias
        out = df.copy()
        if kind == "tick" and "price" not in out.columns and "close" in out.columns:
            out["price"] = out["close"]
        cols = [c for c in keep if c in out.columns]
        return out[cols] if cols else out

    def save(self, df: pd.DataFrame, symbol: str, date: str, kind: str = "1s") -> Path:
        path = self.path_for(symbol, date, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        if df is None or df.empty:
            logger.warning(f"skip empty save {path}")
            return path
        slim = self._slim(df, kind)
        try:
            slim.to_parquet(path, index=False, compression=self.compression)
        except Exception:
            slim.to_parquet(path, index=False, compression="snappy")
        logger.info(f"saved {len(slim)} rows -> {path}")
        return path

    def load(self, symbol: str, date: str, kind: str = "1s") -> pd.DataFrame:
        path = self.path_for(symbol, date, kind)
        if not path.exists() and symbol.upper() == "TMF":
            legacy = self._legacy_path(symbol, date, kind)
            if legacy.exists():
                path = legacy
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    def has(self, symbol: str, date: str, kind: str = "1s") -> bool:
        df = self.load(symbol, date, kind)
        return df is not None and not df.empty

    def list_dates(self, symbol: str, kind: str = "1s") -> list[str]:
        folders = [self.cache_dir / symbol.upper() / kind]
        if symbol.upper() == "TMF":
            folders.append(self._legacy_tmf_root / "TMF" / kind)
        dates: set[str] = set()
        for folder in folders:
            if folder.exists():
                dates.update(p.stem for p in folder.glob("*.parquet"))
        return sorted(dates)
