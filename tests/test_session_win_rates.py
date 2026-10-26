#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""日／夜勝率必須分開，禁止混算當過關依據。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestResult
from src.backtest.metrics import (
    pair_round_trips,
    passes_session_daily_win_rate,
    session_win_stats,
    summarize,
)
from src.data.sessions import label_session, trading_day_for


class TestSessionWinRates(unittest.TestCase):
    def test_trading_day_night_evening_rolls(self) -> None:
        self.assertEqual(label_session("2026-04-01 16:00:00"), "night")
        self.assertEqual(trading_day_for("2026-04-01 16:00:00"), "2026-04-02")
        self.assertEqual(trading_day_for("2026-04-02 01:00:00"), "2026-04-02")
        self.assertEqual(trading_day_for("2026-04-02 09:00:00"), "2026-04-02")

    def test_day_and_night_not_mixed(self) -> None:
        trades = [
            # day win
            {"ts": "2026-04-01 09:00:00", "action": "BUY", "price": 100.0, "reason": "d"},
            {
                "ts": "2026-04-01 10:00:00",
                "action": "CLOSE",
                "price": 110.0,
                "pnl": 100.0,
                "reason": "tp",
            },
            # night loss same calendar afternoon→next trading day
            {"ts": "2026-04-01 16:00:00", "action": "BUY", "price": 100.0, "reason": "n"},
            {
                "ts": "2026-04-01 17:00:00",
                "action": "CLOSE",
                "price": 90.0,
                "pnl": -100.0,
                "reason": "sl",
            },
        ]
        trips = pair_round_trips(trades)
        self.assertEqual(trips[0]["session"], "day")
        self.assertEqual(trips[1]["session"], "night")
        stats = session_win_stats(trips)
        self.assertEqual(stats["day"]["daily_win_rate"], 1.0)
        self.assertEqual(stats["night"]["daily_win_rate"], 0.0)
        # mixed would be 0.5 if same exit calendar — must not be used as day rate
        mixed_like = 0.5
        self.assertNotEqual(stats["day"]["daily_win_rate"], mixed_like)

        result = BacktestResult(trades=trades, final_equity=10000.0, equity_curve=[])
        s = summarize(result, capital=10000.0, label="t")
        self.assertEqual(s["daily_win_rate"], 1.0)  # day only
        self.assertEqual(s["night_daily_win_rate"], 0.0)
        self.assertFalse(
            passes_session_daily_win_rate(s["by_session"], 0.7, require_sessions=["day", "night"])
        )
        self.assertTrue(
            passes_session_daily_win_rate(s["by_session"], 0.7, require_sessions=["day"])
        )


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSessionWinRates)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
