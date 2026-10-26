#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tidemark 簡單戰情 UI — http://0.0.0.0:8080"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn
import yaml

from src.core.config import ROOT as APP_ROOT
from src.ui.app import create_app


def main() -> int:
    cfg_path = APP_ROOT / "config" / "ops.yaml"
    host, port = "0.0.0.0", 8080
    if cfg_path.exists():
        ops = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        ui = ops.get("ui") or {}
        host = str(ui.get("host", host))
        port = int(ui.get("port", port))
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
