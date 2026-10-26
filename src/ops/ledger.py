"""系統化記錄：status + journal.jsonl + 單日彙總。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import ROOT


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


class OpsLedger:
    def __init__(self, *, status_path: Path, journal_path: Path, days_dir: Path):
        self.status_path = status_path
        self.journal_path = journal_path
        self.days_dir = days_dir
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.days_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def write_status(self, payload: dict[str, Any]) -> None:
        body = {
            "brand": "tidemark",
            "updated_at": self.now_iso(),
            **payload,
        }
        self.status_path.write_text(
            json.dumps(body, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def append_journal(self, kind: str, data: dict[str, Any] | None = None) -> None:
        rec = {
            "ts": self.now_iso(),
            "kind": kind,
            "data": data or {},
        }
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")

    def write_day(self, date: str, payload: dict[str, Any]) -> Path:
        path = self.days_dir / f"{date.replace('-', '')}.json"
        body = {
            "brand": "tidemark",
            "date": date,
            "saved_at": self.now_iso(),
            **payload,
        }
        path.write_text(
            json.dumps(body, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def read_day(self, date: str) -> dict[str, Any]:
        path = self.days_dir / f"{date.replace('-', '')}.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def tail_journal(self, n: int = 100) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        lines = self.journal_path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-max(1, n) :]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out


def ledger_from_ops_cfg(cfg: dict[str, Any]) -> OpsLedger:
    paths = cfg.get("paths") or {}
    return OpsLedger(
        status_path=ROOT / str(paths.get("status", "outputs/ops/status.json")),
        journal_path=ROOT / str(paths.get("journal", "outputs/ops/journal.jsonl")),
        days_dir=ROOT / str(paths.get("days_dir", "outputs/ops/days")),
    )
