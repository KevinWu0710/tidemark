#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.minute_bars import MinuteBarBuilder


class TestMinuteBarBuilder(unittest.TestCase):
    def test_closes_on_minute_roll(self):
        b = MinuteBarBuilder()
        self.assertIsNone(b.on_tick("2026-10-16 09:01:10", 100.0, 1))
        self.assertIsNone(b.on_tick("2026-10-16 09:01:40", 102.0, 2))
        closed = b.on_tick("2026-10-16 09:02:05", 101.0, 1)
        self.assertIsNotNone(closed)
        assert closed is not None
        self.assertEqual(str(closed["ts"]), "2026-10-16 09:01:00")
        self.assertEqual(closed["open"], 100.0)
        self.assertEqual(closed["high"], 102.0)
        self.assertEqual(closed["low"], 100.0)
        self.assertEqual(closed["close"], 102.0)
        self.assertEqual(closed["volume"], 3.0)
        self.assertEqual(closed["session"], "day")

    def test_flush_last(self):
        b = MinuteBarBuilder()
        b.on_tick("2026-10-16 09:03:01", 50.0, 1)
        last = b.flush()
        self.assertIsNotNone(last)
        assert last is not None
        self.assertEqual(last["close"], 50.0)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMinuteBarBuilder)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
