#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""日盤事前過濾：夜盤代理只影響日盤早盤；11:15 後禁進。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy.day_prefilters import DayPrefilterConfig, allow_day_entry, entry_allow_mask


class TestDayPrefilters(unittest.TestCase):
    def _bars(self) -> pd.DataFrame:
        rows = []
        # prior day session
        rows.append(
            {
                "ts": "2026-04-01 13:44:00",
                "open": 20000,
                "high": 20010,
                "low": 19990,
                "close": 20000,
                "volume": 1,
                "session": "day",
            }
        )
        # night dump -150
        rows.append(
            {
                "ts": "2026-04-01 16:00:00",
                "open": 20000,
                "high": 20000,
                "low": 19850,
                "close": 19850,
                "volume": 1,
                "session": "night",
            }
        )
        rows.append(
            {
                "ts": "2026-04-02 01:00:00",
                "open": 19850,
                "high": 19850,
                "low": 19840,
                "close": 19840,
                "volume": 1,
                "session": "night",
            }
        )
        # day open + 30m and + 90m and + 160m
        rows.append(
            {
                "ts": "2026-04-02 09:15:00",
                "open": 19840,
                "high": 19850,
                "low": 19830,
                "close": 19845,
                "volume": 1,
                "session": "day",
            }
        )
        rows.append(
            {
                "ts": "2026-04-02 10:15:00",
                "open": 19845,
                "high": 19860,
                "low": 19840,
                "close": 19850,
                "volume": 1,
                "session": "day",
            }
        )
        rows.append(
            {
                "ts": "2026-04-02 11:30:00",
                "open": 19850,
                "high": 19870,
                "low": 19840,
                "close": 19860,
                "volume": 1,
                "session": "day",
            }
        )
        return pd.DataFrame(rows)

    def test_night_dump_blocks_first_hour(self) -> None:
        cfg = DayPrefilterConfig(enabled=True, night_ret_le=-120, skip_open_mins=60, max_entry_mins=150)
        df = self._bars()
        # at 09:15 hist includes night
        hist = df.iloc[:4]
        ok, why = allow_day_entry(hist, "2026-04-02 09:15:00", cfg)
        self.assertFalse(ok)
        self.assertIn("night_ret", why)
        # after 60m OK if before 11:15
        hist2 = df.iloc[:5]
        ok2, _ = allow_day_entry(hist2, "2026-04-02 10:15:00", cfg)
        self.assertTrue(ok2)
        # after 11:15 blocked
        ok3, why3 = allow_day_entry(df, "2026-04-02 11:30:00", cfg)
        self.assertFalse(ok3)
        self.assertIn("after_", why3)

    def test_mask_aligns(self) -> None:
        cfg = DayPrefilterConfig(enabled=True)
        df = self._bars()
        m = entry_allow_mask(df, cfg)
        # index 3 = 09:15 blocked; 4 = 10:15 ok; 5 = 11:30 blocked
        self.assertFalse(bool(m[3]))
        self.assertTrue(bool(m[4]))
        self.assertFalse(bool(m[5]))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDayPrefilters)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
