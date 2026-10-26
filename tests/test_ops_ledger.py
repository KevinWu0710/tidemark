#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ops.ledger import OpsLedger


class TestOpsLedger(unittest.TestCase):
    def test_status_journal_day(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            led = OpsLedger(
                status_path=base / "status.json",
                journal_path=base / "journal.jsonl",
                days_dir=base / "days",
            )
            led.write_status({"phase": "boot"})
            self.assertEqual(led.read_status().get("phase"), "boot")
            led.append_journal("test", {"x": 1})
            self.assertEqual(led.tail_journal(5)[-1]["kind"], "test")
            p = led.write_day("2026-10-16", {"signal": {"choice": "orb"}})
            self.assertTrue(p.exists())
            self.assertEqual(led.read_day("2026-10-16")["signal"]["choice"], "orb")


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestOpsLedger)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
