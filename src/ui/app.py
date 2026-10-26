"""
tidemark 簡單戰情頁：訊號 | 策略 | 下單（白話，不下單）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.core.config import ROOT, load_settings, notify_prefix
from src.ops.daily_runner import load_ops_config
from src.ops.ledger import ledger_from_ops_cfg
from src.ui.view_model import build_view

TZ = ZoneInfo("Asia/Taipei")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def create_app() -> FastAPI:
    settings = load_settings()
    ops_cfg = load_ops_config()
    ledger = ledger_from_ops_cfg(ops_cfg)
    app = FastAPI(title="tidemark", docs_url="/api/docs")

    def _raw_panels() -> dict[str, Any]:
        status = ledger.read_status()
        live = _read_json(ROOT / "outputs" / "live_tick_latest.json")
        evening = _read_json(ROOT / "outputs" / "ops" / "evening_latest.json")
        date = status.get("date") or datetime.now(TZ).strftime("%Y-%m-%d")
        day = ledger.read_day(str(date))
        # 若今天還沒資料，退回最近覆盤日
        if not day and evening.get("date"):
            date = str(evening.get("date"))
            day = ledger.read_day(date)
        review = (evening.get("review") or day.get("review") or {})
        live_full = _read_json(
            ROOT / "outputs" / f"live_tick_{str(date).replace('-', '')}.json"
        )
        choice = (
            day.get("signal")
            or review.get("signal")
            or live_full.get("choice")
            or live.get("choice")
            or {}
        )
        # day.signal 可能是整包 choice dict
        if isinstance(choice, dict) and "choice" not in choice and choice.get("ok") is None:
            pass
        events = (
            ((day.get("orders") or {}).get("events"))
            or (review.get("orders") or {}).get("events")
            or live_full.get("events")
            or []
        )
        summary = (
            ((day.get("orders") or {}).get("summary"))
            or (review.get("orders") or {}).get("summary")
            or live_full.get("summary")
            or {}
        )
        risk = (
            ((day.get("strategy") or {}).get("risk"))
            or (review.get("strategy") or {}).get("risk")
            or live_full.get("risk")
            or ((choice or {}).get("risk") if isinstance(choice, dict) else {})
            or {}
        )
        return {
            "status": status,
            "date": str(date),
            "choice": choice if isinstance(choice, dict) else {},
            "risk": risk if isinstance(risk, dict) else {},
            "summary": summary if isinstance(summary, dict) else {},
            "events": events if isinstance(events, list) else [],
            "journal": ledger.tail_journal(40),
            "prefix": notify_prefix(settings),
        }

    def _panels() -> dict[str, Any]:
        raw = _raw_panels()
        view = build_view(
            status=raw["status"],
            date=raw["date"],
            choice=raw["choice"],
            risk=raw["risk"],
            summary=raw["summary"],
            events=raw["events"],
            journal=raw["journal"],
            prefix=raw["prefix"],
        )
        view["raw"] = raw  # API 仍可看原始
        return view

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "index.html", _panels())

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        return JSONResponse(_panels())

    @app.get("/api/signal")
    def api_signal() -> JSONResponse:
        return JSONResponse(_panels()["signal"])

    @app.get("/api/strategy")
    def api_strategy() -> JSONResponse:
        return JSONResponse(_panels()["strategy"])

    @app.get("/api/orders")
    def api_orders() -> JSONResponse:
        return JSONResponse(_panels()["orders"])

    @app.get("/api/journal")
    def api_journal() -> JSONResponse:
        return JSONResponse({"items": ledger.tail_journal(200)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "brand": "tidemark"}

    return app
