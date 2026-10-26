#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.ui.app import create_app


class TestUiApp(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("brand"), "tidemark")

    def test_html_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("tidemark", res.text)
        self.assertIn("conn-alert", res.text)
        self.assertIn("連線中斷", res.text)

    def test_api_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("phase_zh", data)
        self.assertIn("signal", data)
        self.assertIn("strategy", data)
        self.assertIn("orders", data)

    def test_api_journal(self):
        res = self.client.get("/api/journal")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("items", data)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestUiApp)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
