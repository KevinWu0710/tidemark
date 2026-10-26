#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""flatten_eod：日盤最後一根強制平倉，不跨夜。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.fast_breakout import FastBreakoutConfig, run_breakout_fast


class TestFlattenEod(unittest.TestCase):
    def test_flattens_on_last_day_bar(self) -> None:
        # day bars then night — position opened mid-day must close at last day
        rows = []
        # 5 day bars: rising so breakout long on bar 3+
        for i in range(5):
            c = 100.0 + i * 2
            rows.append(
                {
                    "ts": pd.Timestamp(f"2026-01-02 09:{i:02d}:00"),
                    "high": c + 0.5,
                    "low": c - 0.5,
                    "close": c,
                    "session": "day",
                }
            )
        # night bar would be dangerous if still in position
        rows.append(
            {
                "ts": pd.Timestamp("2026-01-02 15:00:00"),
                "high": 50.0,
                "low": 40.0,
                "close": 45.0,
                "session": "night",
            }
        )
        bars = pd.DataFrame(rows)
        cfg = FastBreakoutConfig(
            lookback=2,
            sessions=("day",),
            flatten_eod=True,
            slip_ticks=0,
            fee_per_side=0,
            tax_rate=0,
        )
        result = run_breakout_fast(bars, cfg)
        closes = [t for t in result.trades if t["action"] == "CLOSE"]
        self.assertTrue(any(t.get("reason") == "flatten_eod" for t in closes))
        # after all bars, must be flat (no open overnight into night crash)
        opens = sum(1 for t in result.trades if t["action"] in ("BUY", "SELL"))
        close_n = sum(1 for t in result.trades if t["action"] == "CLOSE")
        self.assertEqual(opens, close_n)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFlattenEod)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
