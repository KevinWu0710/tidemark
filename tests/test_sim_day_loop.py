#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_settings
from src.data.bars import load_1m_range
from src.trading.sim_day_loop import SimDayConfig, SimDayLoop


class TestSimDayLoop(unittest.TestCase):
    def test_historical_once_no_connect(self):
        settings = load_settings()
        # 用已知有 cache 的區間末日
        date = "2026-10-22"
        bars = load_1m_range("2026-10-01", date)
        if bars is None or bars.empty:
            self.skipTest("no 1m cache")
        loop = SimDayLoop(
            settings,
            cfg=SimDayConfig(
                date=date, dry_run=True, place_sim=False, connect_sim=False
            ),
        )
        try:
            res = loop.run_once(bars)
            self.assertTrue(res.ok)
            self.assertEqual(res.payload.get("date"), date)
            self.assertTrue(res.payload.get("dry_run"))
            self.assertIn("choice", res.payload)
        finally:
            loop.close()


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSimDayLoop)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
