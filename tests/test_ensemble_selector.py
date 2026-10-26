#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy.ensemble_selector import OpenProbeFeatures, SelectorRule, decide


def _feat(**kw) -> OpenProbeFeatures:
    base = dict(
        date="2026-01-02",
        gap=80.0,
        or_range=50.0,
        gap_unfilled_long=True,
        open_minus_ema=10.0,
        probe_close_pos=0.8,
        probe_vol_mean=100.0,
        prior_day_range=120.0,
    )
    base.update(kw)
    return OpenProbeFeatures(**base)


class TestEnsembleSelector(unittest.TestCase):
    def test_always_trend(self):
        r = SelectorRule(id="a", default_choice="trend")
        self.assertEqual(decide(_feat(), r), "trend")

    def test_gap_ge_secure(self):
        r = SelectorRule(
            id="g",
            gap_ge=60.0,
            then_choice="secure",
            else_choice="trend",
        )
        self.assertEqual(decide(_feat(gap=80), r), "secure")
        self.assertEqual(decide(_feat(gap=40), r), "trend")

    def test_skip_or_wide(self):
        r = SelectorRule(
            id="s",
            skip_or_range_gt=100.0,
            gap_ge=40.0,
            then_choice="secure",
            else_choice="trend",
        )
        self.assertEqual(decide(_feat(or_range=150), r), "flat")

    def test_probe_pos_lt(self):
        r = SelectorRule(
            id="p",
            probe_close_pos_lt=0.5,
            then_choice="secure",
            else_choice="trend",
        )
        self.assertEqual(decide(_feat(probe_close_pos=0.3), r), "secure")
        self.assertEqual(decide(_feat(probe_close_pos=0.7), r), "trend")

    def test_abs_gap_le_rsi(self):
        r = SelectorRule(
            id="ag",
            abs_gap_le=40.0,
            then_choice="rsi",
            else_choice="orb",
        )
        self.assertEqual(decide(_feat(gap=20), r), "rsi")
        self.assertEqual(decide(_feat(gap=80), r), "orb")


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestEnsembleSelector)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
