#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
覆盤／紙上重播：用 cache 1m 走帳本。
預設跟 settings trading.paper_strategy（complementary）；可 --signal tmf_v1。

結束寫 outputs/paper_sessions/{session_id}.json + paper_session_latest.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.fast_signals import (
    FastSignalConfig,
    _long_only_tp_loop,
    rsi_tp_long_entry_mask,
    run_rsi_fast,
    run_rsi_tp_fast,
)
from src.backtest.metrics import pair_round_trips, summarize
from src.core.config import load_settings, notify_prefix
from src.core.types import TradeAction
from src.data.bars import load_1m_range
from src.signals.candidates import build_signal
from src.signals.tmf_v1 import TmfV1Signal
from src.strategy.day_prefilters import DayPrefilterConfig, entry_allow_mask
from src.strategy.tmf_session import TmfSessionStrategy
from src.trading.paper import PaperBroker


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _sanitize_id(tag: str) -> str:
    sid = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)[:80]


def _write_latest(path: Path, summary: dict[str, Any], meta: dict[str, Any]) -> None:
    latest = ROOT / "outputs" / "paper_session_latest.json"
    latest.write_text(
        json.dumps(
            {"status": "DONE", "path": str(path), "summary": summary, "meta": meta},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_fast(
    bars,
    params: dict[str, Any],
    trading: dict[str, Any],
    slip: float,
    session_id: str,
) -> dict[str, Any]:
    cfg = FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=tuple(params.get("sessions") or ["day"]),
        stop_points=float(params.get("stop_points") or 0),
        max_daily_loss=float(params.get("max_daily_loss") or 0),
        flatten_eod=bool(params.get("flatten_eod") or params.get("flatten_off_session")),
        allow_short=bool(params.get("allow_short", False)),
    )
    period = int(params.get("rsi_period", 28))
    lo = float(params.get("rsi_low", 20))
    hi = float(params.get("rsi_high", 99.5))
    tp = float(params.get("take_profit_points") or 0)
    pcfg = DayPrefilterConfig.from_params(params)
    if tp > 0 and pcfg.enabled:
        raw = rsi_tp_long_entry_mask(bars, period, lo)
        allow = entry_allow_mask(bars, pcfg)
        if "session" in bars.columns:
            is_day = bars.reset_index(drop=True)["session"].astype(str).values == "day"
        else:
            is_day = allow
        result = _long_only_tp_loop(
            bars,
            cfg,
            raw & allow & is_day,
            take_profit_points=tp,
            reason_entry="rsi_tp_long_prefilter",
        )
    elif tp > 0:
        result = run_rsi_tp_fast(bars, period, lo, cfg, take_profit_points=tp)
    else:
        result = run_rsi_fast(bars, period, lo, hi, cfg)
    cap = float(trading["capital"])
    s = summarize(result, capital=cap, label="PAPER")
    trips = pair_round_trips(result.trades)
    n_eod = sum(1 for t in result.trades if t.get("reason") == "flatten_eod")
    summary = {
        "session_id": session_id,
        "instrument": trading.get("instrument", "TMF"),
        "capital": cap,
        "cash": round(float(result.final_equity), 2),
        "equity": round(float(result.final_equity), 2),
        "net_pnl": round(float(s["net_pnl"]), 2),
        "realized_pnl": round(float(s["net_pnl"]), 2),
        "unrealized": 0.0,
        "n_fills": len(result.trades),
        "n_round_trips": int(s["n_round_trips"]),
        "n_wins": int(s["n_wins"]),
        "n_losses": int(s["n_losses"]),
        # 頂層 win_rate = 日盤；無成交時為 null（勿 float(None)）
        "win_rate": s.get("win_rate"),
        "day_win_rate": s.get("day_win_rate"),
        "day_daily_win_rate": s.get("day_daily_win_rate"),
        "night_win_rate": s.get("night_win_rate"),
        "night_daily_win_rate": s.get("night_daily_win_rate"),
        "by_session": s.get("by_session"),
        "max_drawdown": round(float(s.get("max_drawdown") or 0), 2),
        "position": "FLAT",
        "lots": 0,
        "n_rejects": 0,
        "n_flatten_eod": n_eod,
        "engine": "fast_rsi",
        "prefilter_enabled": bool(pcfg.enabled),
    }
    return {
        "status": "DONE",
        "brand": "tidemark",
        "kind": "paper_session",
        "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary": summary,
        "fills": result.trades,
        "round_trips": trips,
        "rejects": [],
        "equity_marks": [],
    }


def _run_slow(
    bars,
    params: dict[str, Any],
    trading: dict[str, Any],
    settings: dict[str, Any],
    signal_name: str,
    session_id: str,
    mark_every: int,
) -> tuple[PaperBroker, float, int]:
    if signal_name == "tmf_v1":
        signal = TmfV1Signal(params=params)
    else:
        signal = build_signal(
            signal_name,
            {
                "lookback": params.get("lookback", 60),
                "ema_period": params.get("ema_period", 20),
                "rsi_period": params.get("rsi_period", 28),
                "rsi_low": params.get("rsi_low", 15),
                "rsi_high": params.get("rsi_high", 85),
                "sessions": params.get("sessions", ["day"]),
                "allow_short": params.get("allow_short", False),
                "flatten_off_session": params.get("flatten_off_session", True),
            },
        )
    strategy = TmfSessionStrategy(
        signal=signal,
        max_lots=int(trading["max_lots"]),
        capital=float(trading["capital"]),
        params=params,
        stop_points=float(params.get("stop_points", 0) or 0),
        max_daily_loss=float(params.get("max_daily_loss", 0) or 0),
    )
    paper = PaperBroker.from_settings(settings, session_id=session_id)
    df = bars.reset_index(drop=True)
    last_mid = float(df.iloc[-1]["close"])
    for i in range(len(df)):
        hist = df.iloc[: i + 1]
        row = df.iloc[i]
        now_ts = row["ts"] if "ts" in df.columns else i
        mid = float(row["close"])
        last_mid = mid
        decision = strategy.on_bar(hist, now_ts=now_ts)
        if decision.action != TradeAction.HOLD:
            ok = paper.place(decision.action, mid, decision.reason, ts=now_ts)
            if ok:
                if decision.action == TradeAction.CLOSE:
                    strategy.apply_fill(
                        decision.action, pnl=paper.round_trips[-1]["net_pnl"]
                    )
                else:
                    strategy.apply_fill(
                        decision.action, lots=1, price=paper.entry_price
                    )
        if mark_every and (i % mark_every == 0 or i == len(df) - 1):
            paper.mark(mid, ts=now_ts)
    return paper, last_mid, len(df)


def _run_complementary_session(
    bars,
    trading: dict[str, Any],
    slip: float,
    session_id: str,
) -> dict[str, Any]:
    from src.backtest.complementary_fast import ComplementaryConfig, run_complementary_fast

    cfg = ComplementaryConfig.from_locked(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=slip,
    )
    result, cmeta = run_complementary_fast(bars, cfg)
    cap = float(trading["capital"])
    s = summarize(result, capital=cap, label="PAPER")
    trips = pair_round_trips(result.trades)
    summary = {
        "session_id": session_id,
        "instrument": trading.get("instrument", "TMF"),
        "capital": cap,
        "cash": round(float(result.final_equity), 2),
        "equity": round(float(result.final_equity), 2),
        "net_pnl": round(float(s["net_pnl"]), 2),
        "n_fills": len(result.trades),
        "n_round_trips": int(s["n_round_trips"]),
        "n_wins": int(s["n_wins"]),
        "n_losses": int(s["n_losses"]),
        "win_rate": s.get("win_rate"),
        "day_win_rate": s.get("day_win_rate"),
        "day_daily_win_rate": s.get("day_daily_win_rate"),
        "daily_win_rate": s.get("daily_win_rate"),
        "max_drawdown": round(float(s.get("max_drawdown") or 0), 2),
        "engine": "complementary_orb_rsi",
        "rule_id": cfg.rule.id,
        "risk": cfg.risk_dict(),
        "n_choice_orb": cmeta.get("n_choice_orb"),
        "n_choice_rsi": cmeta.get("n_choice_rsi"),
        "n_choice_flat": cmeta.get("n_choice_flat"),
        "n_cooldown_flat": cmeta.get("n_cooldown_flat"),
    }
    return {
        "status": "DONE",
        "brand": "tidemark",
        "kind": "paper_session",
        "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary": summary,
        "fills": result.trades,
        "round_trips": trips,
        "rejects": [],
        "equity_marks": [],
        "complementary_meta": cmeta,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark paper replay / 覆盤")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--tag", default="")
    p.add_argument(
        "--signal",
        default="",
        help="tmf_v1 | complementary；預設跟 settings trading.paper_strategy",
    )
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--mark-every", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="快引擎（預設 on）；--no-fast 用慢路徑",
    )
    p.add_argument(
        "--prefilter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆寫日盤事前過濾；預設跟 settings prefilter_enabled",
    )
    args = p.parse_args()

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    paper_strategy = str(trading.get("paper_strategy") or "complementary").strip().lower()
    signal = (args.signal or paper_strategy).strip().lower()
    if signal in ("comp", "orb_rsi"):
        signal = "complementary"
    if signal in ("rsi", "rsi28"):
        signal = "tmf_v1"

    strat_cfg = next(
        (s for s in settings.get("strategies", []) if s.get("id") == "tmf_v1"),
        {},
    )
    params = dict(strat_cfg.get("params") or {})
    if args.lookback is not None:
        params["lookback"] = args.lookback
    if args.prefilter is not None:
        params["prefilter_enabled"] = bool(args.prefilter)

    tag = args.tag or f"{args.start}_{args.end}"
    session_id = _sanitize_id(tag)

    logger.info(
        f"{notify_prefix()} paper replay {args.start}..{args.end} "
        f"signal={signal} fast={args.fast}"
    )
    running = ROOT / "outputs" / "paper_session_latest.json"
    running.parent.mkdir(parents=True, exist_ok=True)
    running.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "bar_range": [args.start, args.end],
                "tag": tag,
                "fast": args.fast,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bars = load_1m_range(args.start, args.end)
    if bars is None or bars.empty:
        logger.error(f"{notify_prefix()} no bars")
        return 1

    meta = {
        "tag": tag,
        "signal": signal,
        "paper_strategy": paper_strategy,
        "params": {
            k: params.get(k)
            for k in (
                "inner",
                "rsi_period",
                "rsi_low",
                "rsi_high",
                "sessions",
                "allow_short",
                "flatten_eod",
                "flatten_off_session",
                "take_profit_points",
                "stop_points",
                "max_daily_loss",
            )
            if k in params
        },
        "bar_range": [args.start, args.end],
        "n_bars": len(bars),
        "fast": args.fast,
    }

    out_dir = ROOT / "outputs" / "paper_sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else out_dir / f"{session_id}.json"

    if signal == "complementary":
        payload = _run_complementary_session(bars, trading, slip, session_id)
        meta["risk"] = payload["summary"].get("risk")
        payload["meta"] = meta
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        summary = payload["summary"]
        _write_latest(path, summary, meta)
    elif args.fast:
        payload = _run_fast(bars, params, trading, slip, session_id)
        payload["meta"] = meta
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        summary = payload["summary"]
        _write_latest(path, summary, meta)
    else:
        paper, last_mid, n = _run_slow(
            bars, params, trading, settings, "tmf_v1", session_id, args.mark_every
        )
        meta["n_bars"] = n
        path = paper.save_session(
            path,
            meta=meta,
            mid=last_mid,
            include_marks=bool(args.mark_every),
            root=out_dir,
        )
        summary = paper.summary(last_mid)
        _write_latest(path, summary, meta)

    logger.info(
        f"{notify_prefix()} replay done net={summary.get('net_pnl')} "
        f"trips={summary.get('n_round_trips')} -> {path}"
    )
    print(
        json.dumps(
            {"status": "DONE", "path": str(path), "summary": summary, "meta": meta},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
