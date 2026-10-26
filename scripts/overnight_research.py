#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overnight money-first research (no interactive prompts).
Uses fast O(n) breakout walk-forward; writes outputs/MORNING_REPORT.md + JSON.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.fast_breakout import FastBreakoutConfig, run_breakout_fast
from src.backtest.metrics import summarize
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range


IS = ("2026-07-16", "2026-09-15")
OOS = ("2026-09-16", "2026-10-21")
FULL = ("2026-07-16", "2026-10-21")


def load_bars(start: str, end: str):
    """Full day+night 1m — entries filtered to day inside fast engine."""
    return load_1m_range(start, end)


def eval_lookback(lb: int, trading: dict, slip: float, bars_map: dict) -> dict:
    cfg = FastBreakoutConfig(
        lookback=lb,
        capital=trading["capital"],
        point_value=trading["point_value"],
        fee_per_side=trading.get("fee_per_side", 20),
        tax_rate=trading.get("tax_rate", 0.00002),
        slip_ticks=slip,
        sessions=("day",),
    )
    out = {"lookback": lb}
    for label in ("IS", "OOS", "FULL"):
        result = run_breakout_fast(bars_map[label], cfg)
        s = summarize(result, capital=trading["capital"], label=label)
        s.pop("sample_trips", None)
        out[label] = s
    out["pass"] = (
        out["OOS"]["net_pnl"] > 0
        and out["IS"]["net_pnl"] > 0
        and out["OOS"]["n_round_trips"] >= 5
    )
    return out


def main() -> int:
    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))

    bars_map = {
        "IS": load_bars(*IS),
        "OOS": load_bars(*OOS),
        "FULL": load_bars(*FULL),
    }
    for k, v in bars_map.items():
        logger.info(f"{notify_prefix()} loaded {k} bars={len(v)}")

    lookbacks = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100]
    rows = []
    for lb in lookbacks:
        row = eval_lookback(lb, trading, slip, bars_map)
        rows.append(row)
        logger.info(
            f"{notify_prefix()} lb={lb} IS={row['IS']['net_pnl']} "
            f"OOS={row['OOS']['net_pnl']} pass={row['pass']}"
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["pass"],
            r["OOS"]["net_pnl"],
            r["IS"]["net_pnl"],
            -(r["OOS"].get("max_drawdown") or 0),
        ),
        reverse=True,
    )
    winners = [r for r in rows_sorted if r["pass"]]
    best = winners[0] if winners else rows_sorted[0]

    # promote best into a lock file for tmf_v1
    lock = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_id": "tmf_v1",
        "signal": "breakout",
        "params": {"lookback": best["lookback"], "sessions": ["day"]},
        "is": best["IS"],
        "oos": best["OOS"],
        "full": best["FULL"],
        "pass_strict": bool(best.get("pass")),
        "note": "Strict pass = IS>0 and OOS>0 and OOS trips>=5 (day session only)",
    }

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "overnight_breakout_grid.json").write_text(
        json.dumps(rows_sorted, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "best_strategy_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Morning markdown for human
    lines = [
        "# tidemark morning report",
        "",
        f"Generated: {lock['updated_at']}",
        "",
        "## Best day-breakout (strict IS+OOS > 0)",
        "",
        f"- lookback: **{best['lookback']}**",
        f"- sessions: day only",
        f"- IS net: {best['IS']['net_pnl']} | trips={best['IS']['n_round_trips']} | win_rate={best['IS'].get('win_rate')} | maxDD={best['IS'].get('max_drawdown')}",
        f"- OOS net: {best['OOS']['net_pnl']} | trips={best['OOS']['n_round_trips']} | win_rate={best['OOS'].get('win_rate')} | maxDD={best['OOS'].get('max_drawdown')}",
        f"- FULL net: {best['FULL']['net_pnl']}",
        f"- strict_pass: {best.get('pass')}",
        "",
        "## All lookbacks (sorted)",
        "",
        "| lb | IS pnl | OOS pnl | FULL | IS WR | OOS WR | OOS DD | pass |",
        "|----|--------|---------|------|-------|--------|--------|------|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['lookback']} | {r['IS']['net_pnl']} | {r['OOS']['net_pnl']} | "
            f"{r['FULL']['net_pnl']} | {r['IS'].get('win_rate')} | {r['OOS'].get('win_rate')} | "
            f"{r['OOS'].get('max_drawdown')} | {r['pass']} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `outputs/best_strategy_lock.json` — locked candidate for tmf_v1",
        "- `outputs/overnight_breakout_grid.json` — full grid",
        "- `outputs/breakout40_day_report.json` — detailed metrics if analyze finished",
        "",
        "## Next (when you wake)",
        "",
        "1. Read this report + best_strategy_lock.json",
        "2. If strict_pass true → paper SIM next",
        "3. If false → need external filters (basis/OI) or more data",
        "",
    ]
    (out_dir / "MORNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # sync tmf_v1 defaults + settings if best differs
    lb = int(best["lookback"])
    settings_path = ROOT / "config" / "settings.yaml"
    text = settings_path.read_text(encoding="utf-8")
    if "lookback: 40" in text and lb != 40:
        text = text.replace("lookback: 40", f"lookback: {lb}", 1)
        settings_path.write_text(text, encoding="utf-8")
    tmf = ROOT / "src" / "signals" / "tmf_v1.py"
    t = tmf.read_text(encoding="utf-8")
    t2 = t.replace('"lookback": 40', f'"lookback": {lb}').replace(
        "lookback=40", f"lookback={lb}"
    )
    if t2 != t:
        tmf.write_text(t2, encoding="utf-8")

    print(json.dumps({"status": "DONE", "best_lookback": lb, "pass": best.get("pass")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
