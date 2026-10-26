#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""訊號／策略模組邊界：complementary 不混券商。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.complementary_fast import ComplementaryConfig
from src.core.types import TradeAction
from src.signals.complementary import ComplementarySignal
from src.signals.intents import TradeIntent
from src.strategy.complementary_risk import ComplementaryRiskStrategy


class TestComplementaryModules(unittest.TestCase):
    def test_signal_has_no_broker_attrs(self):
        sig = ComplementarySignal(ComplementaryConfig.from_locked())
        self.assertFalse(hasattr(sig, "paper"))
        self.assertFalse(hasattr(sig, "trader"))
        self.assertFalse(hasattr(sig, "place"))

    def test_risk_filters_completed_day(self):
        risk = ComplementaryRiskStrategy(risk=ComplementaryConfig.from_locked().risk_dict())
        risk.note_exit(date="2026-01-02")
        intent = TradeIntent(
            ts=pd.Timestamp("2026-01-02 10:00:00"),
            action=TradeAction.BUY,
            price=20000.0,
            reason="orb_break_high",
            track="orb",
        )
        self.assertFalse(
            risk.allow_bar_intent(intent, date="2026-01-02", session_end=False)
        )

    def test_risk_tick_stop(self):
        risk = ComplementaryRiskStrategy(
            risk=ComplementaryConfig.from_locked().risk_dict(),
            risk_check_seconds=10,
        )
        t0 = pd.Timestamp("2026-01-02 09:10:00")
        risk.note_entry(fill_price=20001.0, ts=t0, track="orb")
        self.assertIsNone(risk.check_tick("2026-01-02 09:10:05", 19940.0, lots=1))
        ex = risk.check_tick("2026-01-02 09:10:10", 19940.0, lots=1)
        self.assertIsNotNone(ex)
        assert ex is not None
        self.assertEqual(ex.reason, "stop_loss")


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestComplementaryModules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
