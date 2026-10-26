#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PaperBroker 帳本：口數、成本、回合賺賠、覆盤落盤。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.types import Side, TradeAction
from src.trading.paper import PaperBroker
from src.trading.sinopac import SinopacFuturesTrader


class TestPaperLedger(unittest.TestCase):
    def test_round_trip_long_pnl(self) -> None:
        b = PaperBroker(
            capital=10000,
            max_lots=1,
            point_value=10,
            fee_per_side=20,
            tax_rate=0.0,
            slip_ticks=0,
            session_id="test_long",
        )
        self.assertTrue(b.place(TradeAction.BUY, 100.0, "enter", ts="t0"))
        self.assertEqual(b.position, Side.LONG)
        self.assertEqual(b.lots, 1)
        # +10 points * 10 = +100 gross; close fee 20
        self.assertTrue(b.place(TradeAction.CLOSE, 110.0, "exit", ts="t1"))
        self.assertEqual(b.position, Side.FLAT)
        self.assertEqual(len(b.round_trips), 1)
        self.assertAlmostEqual(b.round_trips[0]["net_pnl"], 80.0, places=2)
        self.assertAlmostEqual(b.summary()["net_pnl"], b.cash - 10000, places=2)

    def test_max_lots_reject(self) -> None:
        b = PaperBroker(capital=10000, max_lots=1, slip_ticks=0, fee_per_side=0, tax_rate=0)
        self.assertTrue(b.place(TradeAction.BUY, 100.0, "a"))
        self.assertFalse(b.place(TradeAction.BUY, 101.0, "b"))
        self.assertEqual(len(b.rejects), 1)
        self.assertEqual(b.lots, 1)

    def test_save_session(self) -> None:
        b = PaperBroker(
            capital=10000,
            slip_ticks=0,
            fee_per_side=0,
            tax_rate=0,
            session_id="review_demo",
        )
        b.place(TradeAction.SELL, 200.0, "short", ts="a")
        b.place(TradeAction.CLOSE, 190.0, "cover", ts="b")
        with tempfile.TemporaryDirectory() as td:
            path = b.save_session(
                Path(td) / "review_demo.json",
                meta={"tag": "unit"},
                mid=190.0,
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "DONE")
            self.assertEqual(data["kind"], "paper_session")
            self.assertEqual(data["summary"]["n_round_trips"], 1)
            self.assertGreater(data["summary"]["net_pnl"], 0)
            self.assertEqual(data["meta"]["tag"], "unit")


class TestSinopacDryRun(unittest.TestCase):
    def test_max_lots_dry_run(self) -> None:
        t = SinopacFuturesTrader(simulation=True, max_lots=1)
        self.assertTrue(t.place_tmf(TradeAction.BUY, 20000.0, reason="a", dry_run=True))
        self.assertFalse(t.place_tmf(TradeAction.BUY, 20001.0, reason="b", dry_run=True))
        self.assertEqual(t.lots, 1)
        self.assertTrue(t.place_tmf(TradeAction.CLOSE, 20010.0, reason="c", dry_run=True))
        self.assertEqual(t.lots, 0)

    def test_live_blocked_without_allow(self) -> None:
        with self.assertRaises(ValueError):
            SinopacFuturesTrader(simulation=False, allow_live=False)


def main() -> int:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPaperLedger))
    suite.addTests(loader.loadTestsFromTestCase(TestSinopacDryRun))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
