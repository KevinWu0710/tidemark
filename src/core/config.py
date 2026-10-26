from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = ROOT / "config" / "settings.yaml"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_SETTINGS
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    brand = data.setdefault("brand", {})
    brand.setdefault("name", "tidemark")
    brand.setdefault("display_zh", "潮線")
    brand.setdefault("notify_prefix", "[tidemark]")

    trading = data.setdefault("trading", {})
    trading.setdefault("instrument", "TMF")
    trading.setdefault("capital", 10000)
    trading.setdefault("max_lots", 1)
    trading.setdefault("point_value", 10)
    return data


def notify_prefix(settings: dict[str, Any] | None = None) -> str:
    s = settings or load_settings()
    return str(s.get("brand", {}).get("notify_prefix", "[tidemark]"))
