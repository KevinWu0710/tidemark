#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
å¿«è·¯å¾‘ï¼‹ä¸¦è¡Œè¨Šè™ŸæŽƒæï¼ˆbreakout O(n)ï¼‰ã€‚

æ©Ÿåˆ¶ï¼š
1. è³‡æ–™åªä¾ IS/OOS å„è¼‰ä¸€æ¬¡ï¼ˆæ¯ worker é€²ç¨‹è¼‰ä¸€æ¬¡ï¼‰
2. æ¯å€‹åƒæ•¸çµ„åˆç¨ç«‹ç®— â†’ å¯« shard JSONï¼ˆå¯æ–·é»žçºŒè·‘ï¼‰
3. å…¨éƒ¨å®Œæˆå¾Œ merge æŽ’å â†’ signal_scan_report.json

ç”¨æ³•ï¼š
  python scripts/scan_signals_fast.py --workers 4
  python scripts/scan_signals_fast.py --resume          # è·³éŽå·²æœ‰ shard
  python scripts/scan_signals_fast.py --merge-only      # åªåˆä½µæ—¢æœ‰ shard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.fast_breakout import FastBreakoutConfig, run_breakout_fast
from src.backtest.fast_signals import (
    FastSignalConfig,
    run_breakout_long_fast,
    run_ema_fast,
    run_rsi_fast,
    run_rsi_tp_fast,
    run_sma_cross_fast,
)
from src.backtest.metrics import summarize
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range

# per-process bars (Windows spawn: initializer è¼‰å…¥ä¸€æ¬¡)
_IS_BARS = None
_OOS_BARS = None
_TRADING: dict[str, Any] = {}
_SLIP = 1.0


def _combo_id(spec: dict[str, Any]) -> str:
    raw = json.dumps(spec, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _run_id(
    is_range: list[str],
    oos_range: list[str],
    engine: str,
    daytrade: bool,
    grid: str,
) -> str:
    # metrics_v3ï¼šå« daily_win_rateï¼ˆæ—¥å‹çŽ‡ï¼‰
    blob = f"{engine}|{grid}|daytrade={int(daytrade)}|metrics_v3|{is_range}|{oos_range}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def _apply_pass(
    row: dict[str, Any],
    *,
    min_daily_win_rate: float,
    min_oos_trade_days: int,
    min_is_trade_days: int,
    min_oos_trades: int,
    min_is_trades: int,
) -> dict[str, Any]:
    """
    過關：IS/OOS 淨利>0，且日／夜盤「交易日勝率」各自嚴格 > 門檻（禁止混算）。
    有成交的 session 都要過；無成交的 session 略過。
    """
    from src.backtest.metrics import passes_session_daily_win_rate

    is_ok = passes_session_daily_win_rate(
        row.get("is_by_session") or {},
        min_daily_win_rate,
        require_sessions=["day", "night"],
    )
    oos_ok = passes_session_daily_win_rate(
        row.get("oos_by_session") or {},
        min_daily_win_rate,
        require_sessions=["day", "night"],
    )
    # 頂層 is_daily_win_rate 已改為日盤；夜盤見 *_night_daily_win_rate
    row["pass"] = bool(
        float(row.get("is_pnl") or 0) > 0
        and float(row.get("oos_pnl") or 0) > 0
        and int(row.get("oos_trades") or 0) >= min_oos_trades
        and int(row.get("is_trades") or 0) >= min_is_trades
        and int(row.get("is_trade_days") or 0) >= min_is_trade_days
        and int(row.get("oos_trade_days") or 0) >= min_oos_trade_days
        and is_ok
        and oos_ok
    )
    return row


def _dedupe_combos(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in combos:
        cid = _combo_id(c)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out


def combo_grid_daytrade_wide() -> list[dict[str, Any]]:
    """
    æ—¥ç›¤ç•¶æ²–åŠ å¯¬ç¶²æ ¼ï¼ˆé è¶…ç¬¬ä¸€è¼ª 97ï¼‰ï¼š
    åé‡åªåšå¤šï¼ˆéš”å¤œç‰ˆæœ‰éŽé—œçš„æ–¹å‘ï¼‰ï¼‹æ›´å¯† lookbackï¼å‡ç·šï¼RSIï¼åœæã€‚
    """
    sessions = ["day"]
    flat = True
    combos: list[dict[str, Any]] = []
    risk_sets = [
        (0.0, 0.0),
        (100.0, 1000.0),
        (150.0, 1500.0),
        (200.0, 1500.0),
        (250.0, 1500.0),
        (300.0, 2000.0),
        (400.0, 2500.0),
    ]
    lookbacks = list(range(10, 201, 5))  # 10..200
    emas = list(range(5, 201, 5))
    sma_pairs = [
        (a, b)
        for a in (3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50)
        for b in (20, 30, 40, 50, 60, 80, 90, 100, 120, 150, 180, 200)
        if b > a * 1.5
    ]
    rsi_sets = [
        (p, lo, hi)
        for p in (4, 5, 6, 8, 10, 14, 21, 28)
        for lo, hi in ((15, 85), (20, 80), (25, 75), (30, 70), (35, 65))
    ]

    for stop, daily in risk_sets:
        for lb in lookbacks:
            combos.append(
                {
                    "signal": "breakout_long",
                    "params": {"lookback": lb, "sessions": sessions},
                    "stop_points": stop,
                    "max_daily_loss": daily,
                    "flatten_eod": flat,
                }
            )
            # ä»ä¿ç•™å°‘æ•¸é›™å‘ï¼Œé¿å…å®Œå…¨æ”¾æ£„
            if lb % 20 == 0:
                combos.append(
                    {
                        "signal": "breakout",
                        "params": {"lookback": lb, "sessions": sessions},
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                    }
                )
        for ema in emas:
            combos.append(
                {
                    "signal": "ema_mom",
                    "params": {
                        "ema_period": ema,
                        "sessions": sessions,
                        "allow_short": False,
                    },
                    "stop_points": stop,
                    "max_daily_loss": daily,
                    "flatten_eod": flat,
                    "tag": "ema_long_only",
                }
            )
            if ema % 20 == 0:
                combos.append(
                    {
                        "signal": "ema_mom",
                        "params": {"ema_period": ema, "sessions": sessions},
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                    }
                )
        for fast, slow in sma_pairs:
            combos.append(
                {
                    "signal": "sma_cross",
                    "params": {
                        "sma_fast": fast,
                        "sma_slow": slow,
                        "sessions": sessions,
                        "allow_short": False,
                    },
                    "stop_points": stop,
                    "max_daily_loss": daily,
                    "flatten_eod": flat,
                    "tag": "sma_long_only",
                }
            )
        for period, lo, hi in rsi_sets:
            combos.append(
                {
                    "signal": "rsi_mr",
                    "params": {
                        "rsi_period": period,
                        "rsi_low": lo,
                        "rsi_high": hi,
                        "sessions": sessions,
                        "allow_short": False,
                    },
                    "stop_points": stop,
                    "max_daily_loss": daily,
                    "flatten_eod": flat,
                    "tag": "rsi_long_only",
                }
            )
            combos.append(
                {
                    "signal": "rsi_mr",
                    "params": {
                        "rsi_period": period,
                        "rsi_low": lo,
                        "rsi_high": hi,
                        "sessions": sessions,
                    },
                    "stop_points": stop,
                    "max_daily_loss": daily,
                    "flatten_eod": flat,
                }
            )

    return _dedupe_combos(combos)


def combo_grid_wr_hunt(*, daytrade: bool = True) -> list[dict[str, Any]]:
    """
    å°ˆæ”»é«˜å‹çŽ‡ï¼šRSI åªåšå¤š + å›ºå®šåœåˆ©é»žæ•¸ï¼ˆå° TPï¼è¼ƒå¯¬ stopï¼‰+ æ—¥çµã€‚
    """
    sessions = ["day"]
    flat = bool(daytrade)
    combos: list[dict[str, Any]] = []
    # TP å°ã€stop å¯¬ â†’ æ˜“æé«˜å‹çŽ‡ï¼ˆè‹¥å‡å€¼å›žæ­¸æˆç«‹ï¼‰
    tps = (15, 20, 25, 30, 40, 50, 60, 80)
    stops = (80, 100, 120, 150, 200, 250, 300)
    periods = (6, 8, 10, 14, 21, 28)
    lows = (15, 20, 25, 30, 35)
    for period in periods:
        for lo in lows:
            for tp in tps:
                for stop in stops:
                    if stop < tp * 2:
                        continue  # è‡³å°‘ 1:2 é¢¨éšªå ±é…¬åå‘å‹çŽ‡
                    for daily in (0.0, 1500.0, 2000.0):
                        combos.append(
                            {
                                "signal": "rsi_tp",
                                "params": {
                                    "rsi_period": period,
                                    "rsi_low": lo,
                                    "take_profit_points": tp,
                                    "sessions": sessions,
                                    "allow_short": False,
                                },
                                "stop_points": float(stop),
                                "max_daily_loss": daily,
                                "flatten_eod": flat,
                                "tag": "rsi_tp_wr",
                            }
                        )
    return _dedupe_combos(combos)


def combo_grid_all(*, daytrade: bool = False) -> list[dict[str, Any]]:
    """
    å¤šè¨Šè™Ÿç¶²æ ¼ï¼ˆå¿«å¼•æ“Žï¼‰ï¼šbreakout / breakout_long / ema_mom / rsi_mr / sma_crossã€‚
    daytrade â†’ åªæŽƒ day + flatten_eodã€‚
    """
    combos: list[dict[str, Any]] = []
    sessions_list: tuple[list[str], ...] = (["day"],) if daytrade else (["day"], ["night"])
    flat = bool(daytrade)
    risk_sets = [(0.0, 0.0), (250.0, 1500.0)]

    for sessions in sessions_list:
        for stop, daily in risk_sets:
            for lb in (20, 30, 40, 50, 60, 80, 100, 120):
                combos.append(
                    {
                        "signal": "breakout",
                        "params": {"lookback": lb, "sessions": sessions},
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                    }
                )
                combos.append(
                    {
                        "signal": "breakout_long",
                        "params": {"lookback": lb, "sessions": sessions},
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                    }
                )
            for ema in (10, 20, 30, 40, 60, 90, 120):
                combos.append(
                    {
                        "signal": "ema_mom",
                        "params": {"ema_period": ema, "sessions": sessions},
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                    }
                )
                combos.append(
                    {
                        "signal": "ema_mom",
                        "params": {
                            "ema_period": ema,
                            "sessions": sessions,
                            "allow_short": False,
                        },
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                        "tag": "ema_long_only",
                    }
                )
            for period, lo, hi in (
                (6, 20, 80),
                (6, 25, 75),
                (6, 30, 70),
                (14, 30, 70),
                (14, 25, 75),
                (21, 30, 70),
            ):
                combos.append(
                    {
                        "signal": "rsi_mr",
                        "params": {
                            "rsi_period": period,
                            "rsi_low": lo,
                            "rsi_high": hi,
                            "sessions": sessions,
                        },
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                    }
                )
            for fast, slow in ((5, 20), (10, 30), (10, 40), (20, 60), (20, 90), (30, 90)):
                combos.append(
                    {
                        "signal": "sma_cross",
                        "params": {
                            "sma_fast": fast,
                            "sma_slow": slow,
                            "sessions": sessions,
                        },
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                    }
                )
                combos.append(
                    {
                        "signal": "sma_cross",
                        "params": {
                            "sma_fast": fast,
                            "sma_slow": slow,
                            "sessions": sessions,
                            "allow_short": False,
                        },
                        "stop_points": stop,
                        "max_daily_loss": daily,
                        "flatten_eod": flat,
                        "tag": "sma_long_only",
                    }
                )

    combos.append(
        {
            "signal": "breakout",
            "params": {"lookback": 60, "sessions": ["day"]},
            "stop_points": 250.0,
            "max_daily_loss": 1500.0,
            "flatten_eod": flat,
            "tag": "locked_dd",
        }
    )
    return _dedupe_combos(combos)


# backward alias
def combo_grid_breakout(*, daytrade: bool = False) -> list[dict[str, Any]]:
    return combo_grid_all(daytrade=daytrade)


def _init_worker(
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
    trading: dict[str, Any],
    slip: float,
) -> None:
    global _IS_BARS, _OOS_BARS, _TRADING, _SLIP
    _IS_BARS = load_1m_range(is_start, is_end)
    _OOS_BARS = load_1m_range(oos_start, oos_end)
    _TRADING = trading
    _SLIP = slip


def _base_signal_cfg(spec: dict[str, Any]) -> FastSignalConfig:
    params = spec.get("params") or {}
    sessions = tuple(params.get("sessions") or ["day"])
    allow_short = bool(params.get("allow_short", True))
    if (
        spec.get("tag") in ("ema_long_only", "sma_long_only", "rsi_long_only")
        or spec["signal"] == "breakout_long"
    ):
        allow_short = False
    return FastSignalConfig(
        capital=float(_TRADING["capital"]),
        point_value=float(_TRADING["point_value"]),
        fee_per_side=float(_TRADING.get("fee_per_side", 20)),
        tax_rate=float(_TRADING.get("tax_rate", 0.00002)),
        slip_ticks=float(_SLIP),
        sessions=sessions,
        stop_points=float(spec.get("stop_points") or 0),
        max_daily_loss=float(spec.get("max_daily_loss") or 0),
        flatten_eod=bool(spec.get("flatten_eod", False)),
        allow_short=allow_short,
    )


def _run_spec(bars, spec: dict[str, Any]):
    name = spec["signal"]
    params = spec["params"]
    if name == "breakout":
        cfg = FastBreakoutConfig(
            lookback=int(params["lookback"]),
            capital=float(_TRADING["capital"]),
            point_value=float(_TRADING["point_value"]),
            fee_per_side=float(_TRADING.get("fee_per_side", 20)),
            tax_rate=float(_TRADING.get("tax_rate", 0.00002)),
            slip_ticks=float(_SLIP),
            sessions=tuple(params.get("sessions") or ["day"]),
            stop_points=float(spec.get("stop_points") or 0),
            max_daily_loss=float(spec.get("max_daily_loss") or 0),
            flatten_eod=bool(spec.get("flatten_eod", False)),
        )
        return run_breakout_fast(bars, cfg)
    scfg = _base_signal_cfg(spec)
    if name == "breakout_long":
        return run_breakout_long_fast(bars, int(params["lookback"]), scfg)
    if name == "ema_mom":
        return run_ema_fast(bars, int(params["ema_period"]), scfg)
    if name == "rsi_mr":
        return run_rsi_fast(
            bars,
            int(params["rsi_period"]),
            float(params["rsi_low"]),
            float(params["rsi_high"]),
            scfg,
        )
    if name == "rsi_tp":
        return run_rsi_tp_fast(
            bars,
            int(params["rsi_period"]),
            float(params["rsi_low"]),
            scfg,
            take_profit_points=float(params["take_profit_points"]),
        )
    if name == "sma_cross":
        return run_sma_cross_fast(
            bars, int(params["sma_fast"]), int(params["sma_slow"]), scfg
        )
    raise ValueError(f"unknown signal {name}")


def _eval_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Workerï¼šç®—ä¸€çµ„ IS+OOSã€‚"""
    assert _IS_BARS is not None and _OOS_BARS is not None
    flatten_eod = bool(spec.get("flatten_eod", False))
    stop = float(spec.get("stop_points") or 0)
    daily = float(spec.get("max_daily_loss") or 0)
    is_res = _run_spec(_IS_BARS, spec)
    oos_res = _run_spec(_OOS_BARS, spec)
    is_s = summarize(is_res, capital=_TRADING["capital"], label="IS")
    oos_s = summarize(oos_res, capital=_TRADING["capital"], label="OOS")
    is_pnl = float(is_s["net_pnl"])
    oos_pnl = float(oos_s["net_pnl"])
    oos_trades = int(oos_s["n_round_trips"])
    n_eod = sum(1 for t in is_res.trades + oos_res.trades if t.get("reason") == "flatten_eod")
    row = {
        "id": _combo_id(spec),
        "signal": spec["signal"],
        "params": spec["params"],
        "stop_points": stop,
        "max_daily_loss": daily,
        "flatten_eod": flatten_eod,
        "tag": spec.get("tag"),
        "is_pnl": round(is_pnl, 2),
        "is_trades": int(is_s["n_round_trips"]),
        "is_win_rate": is_s.get("win_rate"),
        "is_trade_days": int(is_s.get("n_trade_days") or 0),
        "is_win_days": int(is_s.get("n_win_days") or 0),
        "is_daily_win_rate": is_s.get("daily_win_rate"),  # 日盤
        "is_night_daily_win_rate": is_s.get("night_daily_win_rate"),
        "is_by_session": is_s.get("by_session"),
        "is_maxdd": round(float(is_s.get("max_drawdown") or 0), 2),
        "oos_pnl": round(oos_pnl, 2),
        "oos_trades": oos_trades,
        "oos_win_rate": oos_s.get("win_rate"),
        "oos_trade_days": int(oos_s.get("n_trade_days") or 0),
        "oos_win_days": int(oos_s.get("n_win_days") or 0),
        "oos_daily_win_rate": oos_s.get("daily_win_rate"),  # 日盤
        "oos_night_daily_win_rate": oos_s.get("night_daily_win_rate"),
        "oos_by_session": oos_s.get("by_session"),
        "oos_maxdd": round(float(oos_s.get("max_drawdown") or 0), 2),
        "n_flatten_eod": n_eod,
        # pass 由主進程依 --min-daily-win-rate 重算（日夜分開）
        "pass": False,
    }
    return row


def _shard_path(shard_dir: Path, combo_id: str) -> Path:
    return shard_dir / f"{combo_id}.json"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_shards(shard_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(shard_dir.glob("*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    rows.sort(key=lambda r: (r.get("pass", False), r.get("oos_pnl", 0), r.get("is_pnl", 0)), reverse=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark fast parallel signal scan")
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--resume", action="store_true", help="è·³éŽå·²å­˜åœ¨çš„ shard")
    p.add_argument("--merge-only", action="store_true", help="åªåˆä½µ shardï¼Œä¸ç®—æ–°çš„")
    p.add_argument("--force", action="store_true", help="å¿½ç•¥æ—¢æœ‰ shard é‡ç®—")
    p.add_argument(
        "--daytrade",
        action="store_true",
        help="æ—¥ç›¤ç•¶æ²–ï¼šsession=day ä¸”æ—¥ç›¤æœ€å¾Œä¸€æ ¹ flatten_eodï¼ˆä¸éš”å¤œï¼‰",
    )
    p.add_argument(
        "--grid",
        choices=("default", "wide", "wr_hunt"),
        default="default",
        help="default=åŽŸç¶²æ ¼ï¼›wide=æ—¥ç›¤ç•¶æ²–åŠ å¯¬ï¼›wr_hunt=RSIå°åœåˆ©å°ˆæ”»é«˜å‹çŽ‡",
    )
    p.add_argument(
        "--min-daily-win-rate",
        type=float,
        default=0.0,
        help="æ—¥å‹çŽ‡é–€æª»ï¼šæœ‰äº¤æ˜“æ—¥ä¸­ã€Œç•¶æ—¥æ·¨åˆ©>0ã€å æ¯”é ˆåš´æ ¼å¤§æ–¼æ­¤å€¼ï¼ˆ0.8=è¶…éŽ80%%ï¼‰",
    )
    p.add_argument(
        "--min-win-rate",
        type=float,
        default=None,
        help="ç›¸å®¹èˆŠæ——æ¨™ï¼›ç­‰åŒ --min-daily-win-rate",
    )
    p.add_argument("--min-oos-trades", type=int, default=5)
    p.add_argument("--min-is-trades", type=int, default=5)
    p.add_argument("--min-oos-trade-days", type=int, default=5)
    p.add_argument("--min-is-trade-days", type=int, default=5)
    args = p.parse_args()

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    is_range = [args.is_start, args.is_end]
    oos_range = [args.oos_start, args.oos_end]
    if args.grid in ("wide", "wr_hunt") and not args.daytrade:
        # wr_hunt é è¨­ç•¶æ²–ï¼›è‹¥æœªåŠ  --daytrade è‡ªå‹•é–‹å•Ÿ
        if args.grid == "wr_hunt":
            args.daytrade = True
        else:
            logger.error(f"{notify_prefix()} --grid wide è«‹æ­é… --daytrade")
            return 1
    min_dwr = float(
        args.min_daily_win_rate
        if args.min_win_rate is None
        else args.min_win_rate
    )
    min_oos_tr = int(args.min_oos_trades)
    min_is_tr = int(args.min_is_trades)
    min_oos_days = int(args.min_oos_trade_days)
    min_is_days = int(args.min_is_trade_days)
    # æ—¥å‹çŽ‡â‰¥80% æ™‚è¦æ±‚è¶³å¤ äº¤æ˜“æ—¥ï¼Œé¿å…å°‘æ•¸å¹¾å¤©é‹æ°£
    if min_dwr >= 0.8:
        min_oos_tr = max(min_oos_tr, 10)
        min_is_tr = max(min_is_tr, 10)
        min_oos_days = max(min_oos_days, 15)
        min_is_days = max(min_is_days, 30)
    pass_kwargs = dict(
        min_daily_win_rate=min_dwr,
        min_oos_trade_days=min_oos_days,
        min_is_trade_days=min_is_days,
        min_oos_trades=min_oos_tr,
        min_is_trades=min_is_tr,
    )
    pass_rule = (
        f"IS>0 and OOS>0 and is_trades>={min_is_tr} and oos_trades>={min_oos_tr} "
        f"and is_trade_days>={min_is_days} and oos_trade_days>={min_oos_days} "
        f"and is_daily_wr>{min_dwr} and oos_daily_wr>{min_dwr}"
    )
    engine = "fast_breakout_daytrade" if args.daytrade else "fast_breakout"
    run_id = _run_id(is_range, oos_range, engine, args.daytrade, args.grid)
    shard_dir = ROOT / "outputs" / "scan_shards" / run_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "outputs" / "signal_scan_report.json"
    detail_path = ROOT / "outputs" / "signal_scan.json"

    if args.grid == "wide":
        grid = combo_grid_daytrade_wide()
    elif args.grid == "wr_hunt":
        grid = combo_grid_wr_hunt(daytrade=args.daytrade)
    else:
        grid = combo_grid_all(daytrade=args.daytrade)
    _write_report(
        report_path,
        {
            "status": "RUNNING",
            "brand": "tidemark",
            "engine": engine,
            "daytrade": args.daytrade,
            "grid": args.grid,
            "run_id": run_id,
            "is_range": is_range,
            "oos_range": oos_range,
            "n_combos": len(grid),
            "shard_dir": str(shard_dir),
            "workers": args.workers,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "note": "ä¸¦è¡Œ shardï¼›å®Œæˆå¾Œ status=DONEã€‚è®€æœ¬æª”å‹¿ç›¯ logã€‚",
        },
    )

    if args.merge_only:
        rows = [
            _apply_pass(
                r,
                **pass_kwargs,
            )
            for r in merge_shards(shard_dir)
        ]
        rows.sort(
            key=lambda r: (r.get("pass", False), r.get("oos_pnl", 0), r.get("is_pnl", 0)),
            reverse=True,
        )
        winners = [r for r in rows if r.get("pass")]
        _write_report(detail_path, rows)
        _write_report(
            report_path,
            {
                "status": "DONE",
                "brand": "tidemark",
                "engine": engine,
                "daytrade": args.daytrade,
                "grid": args.grid,
                "run_id": run_id,
                "is_range": is_range,
                "oos_range": oos_range,
                "n_combos": len(rows),
                "n_winners": len(winners),
                "winners": winners[:20],
                "pass_rule": pass_rule,
                "min_daily_win_rate": min_dwr,
                "detail_path": str(detail_path),
                "shard_dir": str(shard_dir),
                "note": "merge-only",
            },
        )
        logger.info(f"{notify_prefix()} merge-only winners={len(winners)}/{len(rows)}")
        return 0

    pending: list[dict[str, Any]] = []
    for spec in grid:
        cid = _combo_id(spec)
        sp = _shard_path(shard_dir, cid)
        if args.force or not (args.resume and sp.exists()):
            pending.append(spec)
        elif args.resume and sp.exists():
            logger.info(f"{notify_prefix()} skip shard {cid}")

    logger.info(
        f"{notify_prefix()} fast scan run_id={run_id} total={len(grid)} "
        f"pending={len(pending)} workers={args.workers}"
    )

    done = len(grid) - len(pending)
    if pending:
        # workers=1ï¼šåŒé€²ç¨‹ï¼ˆé™¤éŒ¯ï¼çœè¨˜æ†¶é«”ï¼‰
        if args.workers <= 1:
            _init_worker(args.is_start, args.is_end, args.oos_start, args.oos_end, trading, slip)
            for spec in pending:
                row = _apply_pass(
                    _eval_one(spec),
                    **pass_kwargs,
                )
                _shard_path(shard_dir, row["id"]).write_text(
                    json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                done += 1
                flag = "PASS" if row["pass"] else "fail"
                logger.info(
                    f"{notify_prefix()} [{flag}] {done}/{len(grid)} "
                    f"{row['signal']} dWR={row.get('oos_daily_win_rate')} "
                    f"IS={row['is_pnl']} OOS={row['oos_pnl']}"
                )
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(
                    args.is_start,
                    args.is_end,
                    args.oos_start,
                    args.oos_end,
                    trading,
                    slip,
                ),
            ) as ex:
                futs = {ex.submit(_eval_one, spec): spec for spec in pending}
                for fut in as_completed(futs):
                    row = _apply_pass(
                        fut.result(),
                        **pass_kwargs,
                    )
                    _shard_path(shard_dir, row["id"]).write_text(
                        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    done += 1
                    flag = "PASS" if row["pass"] else "fail"
                    logger.info(
                        f"{notify_prefix()} [{flag}] {done}/{len(grid)} "
                        f"{row['signal']} stop={row['stop_points']} "
                        f"dWR_is={row.get('is_daily_win_rate')} "
                        f"dWR_oos={row.get('oos_daily_win_rate')} "
                        f"IS={row['is_pnl']} OOS={row['oos_pnl']}"
                    )
                    # é€²åº¦å¯«å…¥ reportï¼ˆå¯è®€ status=RUNNING + n_doneï¼‰
                    if done % 3 == 0 or done == len(grid):
                        _write_report(
                            report_path,
                            {
                                "status": "RUNNING",
                                "brand": "tidemark",
                                "engine": engine,
                                "daytrade": args.daytrade,
                                "run_id": run_id,
                                "is_range": is_range,
                                "oos_range": oos_range,
                                "n_combos": len(grid),
                                "n_done": done,
                                "shard_dir": str(shard_dir),
                                "workers": args.workers,
                                "updated_at": datetime.now().isoformat(timespec="seconds"),
                            },
                        )

    rows = [
        _apply_pass(
            r,
            **pass_kwargs,
        )
        for r in merge_shards(shard_dir)
    ]
    rows.sort(
        key=lambda r: (
            r.get("pass", False),
            r.get("oos_daily_win_rate") or 0,
            r.get("oos_pnl", 0),
            r.get("is_pnl", 0),
        ),
        reverse=True,
    )
    winners = [r for r in rows if r.get("pass")]
    _write_report(detail_path, rows)
    _write_report(
        report_path,
        {
            "status": "DONE",
            "brand": "tidemark",
            "engine": engine,
            "daytrade": args.daytrade,
            "grid": args.grid,
            "run_id": run_id,
            "is_range": is_range,
            "oos_range": oos_range,
            "n_combos": len(rows),
            "n_winners": len(winners),
            "winners": winners[:20],
            "pass_rule": pass_rule,
            "min_daily_win_rate": min_dwr,
            "min_is_trades": min_is_tr,
            "min_oos_trades": min_oos_tr,
            "detail_path": str(detail_path),
            "shard_dir": str(shard_dir),
            "workers": args.workers,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "note": (
                "daytrade=日盤最後一根 flatten_eod；" if args.daytrade else ""
            )
            + f"每日勝率門檻>{min_dwr}（獲利天數/有交易天數）。",
        },
    )
    logger.info(f"{notify_prefix()} DONE winners={len(winners)}/{len(rows)} -> {report_path}")
    for w in winners[:8]:
        logger.info(
            f"  TOP {w['signal']} {w['params']} stop={w['stop_points']} "
            f"dWR={w.get('oos_daily_win_rate')} OOS={w['oos_pnl']} IS={w['is_pnl']}"
        )
    return 0


if __name__ == "__main__":
    # Windows spawn éœ€è¦ freeze_support
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())

