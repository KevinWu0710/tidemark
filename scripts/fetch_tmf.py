#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thin wrapper: TMF fetch → scripts/fetch_history.py --symbol TMF."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    user = sys.argv[1:]
    if not any(a == "--symbol" or a.startswith("--symbol=") for a in user):
        sys.argv = [sys.argv[0], "--symbol", "TMF", *user]
    path = ROOT / "scripts" / "fetch_history.py"
    spec = importlib.util.spec_from_file_location("fetch_history", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main()


if __name__ == "__main__":
    raise SystemExit(main())
