#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.complementary_fast import (
    LOCKED_COMPLEMENTARY_RULE,
    ComplementaryConfig,
    run_complementary_fast,
)
from src.strategy.ensemble_selector import OpenProbeFeatures, decide


class TestComplementaryFast(unittest.TestCase):
    def test_locked_rule_gap(self):
        feat_big = OpenProbeFeatures(
            date="2026-01-02",
            gap=150.0,
            or_range=50.0,
            gap_unfilled_long=True,
            open_minus_ema=10.0,
            probe_close_pos=0.8,
            probe_vol_mean=100.0,
            prior_day_range=120.0,
        )
        feat_small = OpenProbeFeatures(
            date="2026-01-03",
            gap=30.0,
            or_range=50.0,
            gap_unfilled_long=False,
            open_minus_ema=10.0,
            probe_close_pos=0.5,
            probe_vol_mean=100.0,
            prior_day_range=120.0,
        )
        self.assertEqual(decide(feat_big, LOCKED_COMPLEMENTARY_RULE), "orb")
        self.assertEqual(decide(feat_small, LOCKED_COMPLEMENTARY_RULE), "rsi")

    def test_empty_bars(self):
        res, meta = run_complementary_fast(pd.DataFrame(), ComplementaryConfig())
        self.assertEqual(res.final_equity, 10000.0)
        self.assertEqual(meta.get("n_days", 0), 0)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestComplementaryFast)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
