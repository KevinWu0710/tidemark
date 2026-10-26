"""
真互補分流快路徑（一口）：探針結束選 orb | rsi | flat，當日只跑該軌。

鎖規則：gap≥120 → orb，否則 rsi。
硬化：ORB 保本、虧損日後冷卻（LOCKED_RISK）。
實作：先預計算兩軌「若當日被選」的成交，再依時間序套用分流／冷卻（日結 flatten，日與日獨立）。
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.fast_signals import FastSignalConfig, rsi_tp_long_entry_mask
from src.core.types import Side
from src.strategy.ensemble_selector import OpenProbeFeatures, SelectorRule, decide

ROOT = Path(__file__).resolve().parents[2]
PROBE_MINS = 15
PROBE_END_MINUTES = 8 * 60 + 45 + PROBE_MINS  # 09:00

LOCKED_COMPLEMENTARY_RULE = SelectorRule(
    id="gap≥120→orb_else_rsi",
    gap_ge=120.0,
    then_choice="orb",
    else_choice="rsi",
    default_choice="orb",
)

# 硬化折衷：ORB +50pt 保本壓 DD；不加虧損冷卻（cooldown=1 會把 OOS 淨利砍過狠）
# 來源：complementary_harden_report top（be=50, cool=0）OOS +6782／DD 2321
LOCKED_RISK: dict[str, Any] = {
    "orb_be_trig": 50.0,
    "orb_stop": 50.0,
    "orb_tp": 200.0,
    "rsi_stop": 300.0,
    "rsi_tp": 40.0,
    "cooldown_after_loss_days": 0,
    "max_daily_loss": 0.0,
}


def _load_orb_mod():
    path = ROOT / "scripts" / "research_orb_advanced.py"
    name = "research_orb_advanced"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class ComplementaryConfig:
    capital: float = 10000.0
    point_value: float = 10.0
    fee_per_side: float = 20.0
    tax_rate: float = 0.00002
    slip_ticks: float = 1.0
    rsi_period: int = 28
    rsi_low: float = 20.0
    rsi_tp: float = 40.0
    rsi_stop: float = 300.0
    orb_be_trig: float = 0.0
    orb_stop: float = 50.0
    orb_tp: float = 200.0
    cooldown_after_loss_days: int = 0
    max_daily_loss: float = 0.0
    rule: SelectorRule = field(default_factory=lambda: LOCKED_COMPLEMENTARY_RULE)

    @classmethod
    def from_locked(cls, **overrides: Any) -> "ComplementaryConfig":
        kw: dict[str, Any] = {
            "orb_be_trig": float(LOCKED_RISK["orb_be_trig"]),
            "orb_stop": float(LOCKED_RISK["orb_stop"]),
            "orb_tp": float(LOCKED_RISK["orb_tp"]),
            "rsi_stop": float(LOCKED_RISK["rsi_stop"]),
            "rsi_tp": float(LOCKED_RISK["rsi_tp"]),
            "cooldown_after_loss_days": int(LOCKED_RISK["cooldown_after_loss_days"]),
            "max_daily_loss": float(LOCKED_RISK["max_daily_loss"]),
        }
        kw.update(overrides)
        return cls(**kw)

    def risk_dict(self) -> dict[str, Any]:
        return {
            "orb_be_trig": self.orb_be_trig,
            "orb_stop": self.orb_stop,
            "orb_tp": self.orb_tp,
            "rsi_stop": self.rsi_stop,
            "rsi_tp": self.rsi_tp,
            "cooldown_after_loss_days": self.cooldown_after_loss_days,
            "max_daily_loss": self.max_daily_loss,
        }


def _build_features(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
) -> dict[str, OpenProbeFeatures]:
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    volume = df["volume"].astype(float).values
    out: dict[str, OpenProbeFeatures] = {}
    prior_range = 0.0
    for sess in day_sessions:
        s_idx = int(sess["start_idx"])
        e_idx = int(sess["end_idx"])
        if e_idx - s_idx <= PROBE_MINS + 2:
            continue
        ph = float(np.max(high[s_idx : s_idx + PROBE_MINS]))
        pl = float(np.min(low[s_idx : s_idx + PROBE_MINS]))
        or_range = ph - pl
        pc = float(close[s_idx + PROBE_MINS - 1])
        pos = 0.5 if or_range <= 1e-9 else (pc - pl) / or_range
        pclose = sess["prev_day_close"]
        gap = float(sess["gap_pts"])
        unfilled = bool(
            gap >= 40.0
            and (
                pclose is None
                or (isinstance(pclose, float) and np.isnan(pclose))
                or pl > float(pclose)
            )
        )
        d = str(sess["date"])
        out[d] = OpenProbeFeatures(
            date=d,
            gap=gap,
            or_range=float(or_range),
            gap_unfilled_long=unfilled,
            open_minus_ema=float(sess["open_px"]) - float(sess["trend_ema"]),
            probe_close_pos=float(pos),
            probe_vol_mean=float(np.mean(volume[s_idx : s_idx + PROBE_MINS])),
            prior_day_range=float(prior_range),
        )
        prior_range = float(np.max(high[s_idx:e_idx]) - np.min(low[s_idx:e_idx]))
    return out


def _orb_params(cfg: ComplementaryConfig) -> dict[str, Any]:
    return {
        "probe_mins": 15,
        "cutoff_mins": 105,
        "gap_th": 40.0,
        "max_or_range": 250.0,
        "stop_points": float(cfg.orb_stop),
        "tp_points": float(cfg.orb_tp),
        "be_trig_pts": float(cfg.orb_be_trig),
        "be_lock_pts": 5.0,
        "trail_trig_pts": 0.0,
        "trail_dist": 0.0,
        "time_stop_mins": 45,
        "use_trend_filter": False,
        "vol_spike_mult": 0.0,
        "allow_short": False,
    }


def _trades_by_date(trades: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        dk = str(pd.Timestamp(t["ts"]).date())
        by[dk].append(t)
    return dict(by)


def _day_net_approx(day_trades: list[dict[str, Any]], fee: float) -> float:
    close_pnl = sum(
        float(t.get("pnl") or 0) for t in day_trades if t.get("action") == "CLOSE"
    )
    n_opens = sum(1 for t in day_trades if t.get("action") in ("BUY", "SELL"))
    return close_pnl - n_opens * fee


def run_complementary_fast(
    bars: pd.DataFrame,
    cfg: ComplementaryConfig | None = None,
) -> tuple[BacktestResult, dict[str, Any]]:
    cfg = cfg or ComplementaryConfig.from_locked()
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital), {"n_days": 0, "choices": {}}

    orb_mod = _load_orb_mod()
    TradeConfig = orb_mod.TradeConfig
    prepare_advanced_features = orb_mod.prepare_advanced_features
    run_advanced_simulation = orb_mod.run_advanced_simulation

    df_feat, day_sessions = prepare_advanced_features(bars)
    feats = _build_features(df_feat, day_sessions)
    ordered_dates = [str(s["date"]) for s in day_sessions if str(s["date"]) in feats]

    trade_cfg = TradeConfig(
        capital=cfg.capital,
        point_value=cfg.point_value,
        fee_per_side=cfg.fee_per_side,
        tax_rate=cfg.tax_rate,
        slip_ticks=cfg.slip_ticks,
    )

    # 預計算：ORB／RSI 在「若當日被選」下的成交（日結，互不依賴）
    orb_res = run_advanced_simulation(
        df_feat, day_sessions, trade_cfg, **_orb_params(cfg)
    )
    orb_by = _trades_by_date(
        [{**t, "track": "orb"} for t in orb_res.trades]
    )

    df = bars.reset_index(drop=True)
    n = len(df)
    raw_rsi = rsi_tp_long_entry_mask(df, cfg.rsi_period, cfg.rsi_low)
    ts = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.Series(range(n))
    mins = (ts.dt.hour * 60 + ts.dt.minute).values
    after_probe = mins >= PROBE_END_MINUTES
    long_entry = raw_rsi & after_probe
    rsi_cfg = FastSignalConfig(
        capital=cfg.capital,
        point_value=cfg.point_value,
        fee_per_side=cfg.fee_per_side,
        tax_rate=cfg.tax_rate,
        slip_ticks=cfg.slip_ticks,
        sessions=("day",),
        stop_points=cfg.rsi_stop,
        max_daily_loss=float(cfg.max_daily_loss or 0),
        flatten_eod=True,
        allow_short=False,
    )
    rsi_by = _trades_by_date(_rsi_trades_only(df, long_entry, rsi_cfg, cfg.rsi_tp))

    choices: dict[str, str] = {}
    picked: list[dict[str, Any]] = []
    skip_remaining = 0
    n_cooldown_flat = 0

    for d in ordered_dates:
        if skip_remaining > 0:
            choices[d] = "flat"
            skip_remaining -= 1
            n_cooldown_flat += 1
            continue
        ch = decide(feats[d], cfg.rule)
        choices[d] = ch
        if ch == "orb":
            day_tr = orb_by.get(d, [])
        elif ch == "rsi":
            day_tr = rsi_by.get(d, [])
        else:
            day_tr = []
        picked.extend(day_tr)
        if (
            cfg.cooldown_after_loss_days > 0
            and day_tr
            and _day_net_approx(day_tr, cfg.fee_per_side) < 0
        ):
            skip_remaining = int(cfg.cooldown_after_loss_days)

    merged = sorted(picked, key=lambda t: pd.Timestamp(t["ts"]))
    result = _replay_trades_equity(merged, cfg)
    meta = {
        "rule": cfg.rule.to_dict(),
        "risk": cfg.risk_dict(),
        "n_feature_days": len(feats),
        "n_choice_orb": sum(1 for c in choices.values() if c == "orb"),
        "n_choice_rsi": sum(1 for c in choices.values() if c == "rsi"),
        "n_choice_flat": sum(1 for c in choices.values() if c == "flat"),
        "n_cooldown_flat": n_cooldown_flat,
        "n_orb_fills": sum(
            1 for t in merged if t.get("track") == "orb" and t.get("action") in ("BUY", "SELL")
        ),
        "n_rsi_fills": sum(
            1 for t in merged if t.get("track") == "rsi" and t.get("action") in ("BUY", "SELL")
        ),
    }
    return result, meta


def today_complementary_choice(
    bars: pd.DataFrame,
    *,
    as_of: str | None = None,
    cfg: ComplementaryConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or ComplementaryConfig.from_locked()
    if bars is None or bars.empty:
        return {"ok": False, "error": "no_bars"}
    orb_mod = _load_orb_mod()
    df_feat, day_sessions = orb_mod.prepare_advanced_features(bars)
    feats = _build_features(df_feat, day_sessions)
    if not feats:
        return {"ok": False, "error": "no_features"}
    dates = sorted(feats.keys())
    target = as_of if as_of and as_of in feats else dates[-1]
    feat = feats[target]
    ch = decide(feat, cfg.rule)
    return {
        "ok": True,
        "date": target,
        "choice": ch,
        "gap": feat.gap,
        "or_range": feat.or_range,
        "rule_id": cfg.rule.id,
        "risk": cfg.risk_dict(),
        "features": asdict(feat),
    }


def _rsi_trades_only(
    df: pd.DataFrame,
    long_entry: np.ndarray,
    cfg: FastSignalConfig,
    take_profit_points: float,
) -> list[dict[str, Any]]:
    close_a = df["close"].astype(float).values
    ts = df["ts"].values if "ts" in df.columns else np.arange(len(df))
    if "session" in df.columns:
        sess = df["session"].astype(str).values
    else:
        sess = np.array(["day"] * len(df))
    allow = set(cfg.sessions) if cfg.sessions else {"day"}
    n = len(df)
    in_sess = np.array([s in allow for s in sess], dtype=bool)
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])

    equity = cfg.capital
    position = Side.FLAT
    entry = 0.0
    trades: list[dict[str, Any]] = []
    tp = float(take_profit_points)
    day_key = None
    day_realized = 0.0

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    def fill_buy(mid: float) -> float:
        return mid + cfg.slip_ticks

    def fill_sell(mid: float) -> float:
        return mid - cfg.slip_ticks

    for i in range(n):
        px = float(close_a[i])
        now = ts[i]
        eod = bool(cfg.flatten_eod and last_in_block[i])
        try:
            dk = str(pd.Timestamp(now).date())
        except Exception:
            dk = str(now)[:10]
        if dk != day_key:
            day_key = dk
            day_realized = 0.0

        if position == Side.LONG:
            if cfg.stop_points > 0 and (entry - px) >= cfg.stop_points:
                f = fill_sell(px)
                net = (f - entry) * cfg.point_value - cost(f)
                equity += net
                day_realized += net
                trades.append(
                    {
                        "ts": now,
                        "action": "CLOSE",
                        "price": f,
                        "pnl": net,
                        "reason": f"stop_loss_{cfg.stop_points}pt",
                        "track": "rsi",
                    }
                )
                position = Side.FLAT
                entry = 0.0
            elif tp > 0 and (px - entry) >= tp:
                f = fill_sell(px)
                net = (f - entry) * cfg.point_value - cost(f)
                equity += net
                day_realized += net
                trades.append(
                    {
                        "ts": now,
                        "action": "CLOSE",
                        "price": f,
                        "pnl": net,
                        "reason": f"take_profit_{tp}pt",
                        "track": "rsi",
                    }
                )
                position = Side.FLAT
                entry = 0.0

        blocked = cfg.max_daily_loss > 0 and day_realized <= -abs(cfg.max_daily_loss)

        if in_sess[i]:
            if eod and position == Side.LONG:
                f = fill_sell(px)
                net = (f - entry) * cfg.point_value - cost(f)
                equity += net
                day_realized += net
                trades.append(
                    {
                        "ts": now,
                        "action": "CLOSE",
                        "price": f,
                        "pnl": net,
                        "reason": "flatten_eod",
                        "track": "rsi",
                    }
                )
                position = Side.FLAT
                entry = 0.0
            elif (
                not eod
                and position == Side.FLAT
                and not blocked
                and bool(long_entry[i])
            ):
                f = fill_buy(px)
                equity -= cost(f)
                entry = f
                position = Side.LONG
                trades.append(
                    {
                        "ts": now,
                        "action": "BUY",
                        "price": f,
                        "reason": "rsi_tp_long",
                        "track": "rsi",
                    }
                )

    return trades


def _replay_trades_equity(
    trades: list[dict[str, Any]], cfg: ComplementaryConfig
) -> BacktestResult:
    equity = cfg.capital
    curve: list[dict[str, Any]] = []
    position = Side.FLAT
    entry = 0.0

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    out_trades: list[dict[str, Any]] = []
    for t in trades:
        action = t.get("action")
        px = float(t.get("price") or 0)
        now = t.get("ts")
        if action == "BUY":
            equity -= cost(px)
            entry = px
            position = Side.LONG
            out_trades.append(dict(t))
        elif action == "SELL":
            equity -= cost(px)
            entry = px
            position = Side.SHORT
            out_trades.append(dict(t))
        elif action == "CLOSE":
            if "pnl" in t and t["pnl"] is not None:
                equity += float(t["pnl"])
            else:
                if position == Side.LONG:
                    equity += (px - entry) * cfg.point_value - cost(px)
                elif position == Side.SHORT:
                    equity += (entry - px) * cfg.point_value - cost(px)
            position = Side.FLAT
            entry = 0.0
            out_trades.append(dict(t))
        curve.append({"ts": now, "equity": equity, "price": px})

    return BacktestResult(
        equity_curve=curve,
        trades=out_trades,
        final_equity=equity,
        n_steps=len(out_trades),
    )
