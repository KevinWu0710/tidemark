"""
永豐 Shioaji futures adapter — SIM first.

Hard rules:
- default simulation=True (LIVE requires explicit allow_live=True + env keys)
- quantity / open position capped at max_lots=1
- prefer PaperBroker for historical replay; this class is for SIM order path
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger

from src.core.config import ROOT, notify_prefix
from src.core.types import Side, TradeAction

load_dotenv(ROOT / ".env")


@dataclass
class SinopacFuturesTrader:
    """
    Thin Shioaji futures order wrapper for TMF (near-month continuous TMFR1).

    Local lot tracking enforces max_lots even if the broker would allow more.
    """

    simulation: bool = True
    allow_live: bool = False
    max_lots: int = 1
    instrument: str = "TMF"
    contract_code: str = "TMFR1"

    api: Any = field(default=None, repr=False)
    position: Side = Side.FLAT
    lots: int = 0
    last_trade: Any = field(default=None, repr=False)
    fills: list[dict[str, Any]] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_lots = int(self.max_lots)
        if self.max_lots != 1:
            logger.warning(
                f"{notify_prefix()} SinopacFuturesTrader forcing max_lots=1 "
                f"(requested {self.max_lots})"
            )
            self.max_lots = 1
        if not self.simulation and not self.allow_live:
            raise ValueError(
                "LIVE trading blocked: pass allow_live=True only after explicit user order"
            )

    def _keys(self) -> tuple[str, str]:
        if self.simulation:
            return (
                os.environ.get("SHIOAJI_SIM_API_KEY", ""),
                os.environ.get("SHIOAJI_SIM_SECRET_KEY", ""),
            )
        return (
            os.environ.get("SHIOAJI_LIVE_API_KEY", ""),
            os.environ.get("SHIOAJI_LIVE_SECRET_KEY", ""),
        )

    def connect(self) -> bool:
        if self.api is not None:
            return True
        key, secret = self._keys()
        if not key or not secret:
            logger.error(f"{notify_prefix()} Shioaji keys missing in .env (sim={self.simulation})")
            return False
        try:
            import shioaji as sj

            self.api = sj.Shioaji(simulation=self.simulation)
            self.api.login(api_key=key, secret_key=secret)
            self._maybe_activate_ca()
            logger.info(
                f"{notify_prefix()} SinopacFuturesTrader connected sim={self.simulation}"
            )
            return True
        except Exception as e:
            logger.error(f"{notify_prefix()} Sinopac connect failed: {e}")
            self.api = None
            return False

    def _maybe_activate_ca(self) -> None:
        """CA optional on SIM; required for some LIVE flows."""
        assert self.api is not None
        ca_path = os.environ.get("SHIOAJI_CERT_PATH", "")
        ca_passwd = os.environ.get("SHIOAJI_CERT_PASSWORD", "")
        person_id = os.environ.get("SHIOAJI_PERSON_ID", "")
        if not ca_path or not ca_passwd:
            return
        try:
            kwargs: dict[str, Any] = {"ca_path": ca_path, "ca_passwd": ca_passwd}
            if person_id:
                kwargs["person_id"] = person_id
            self.api.activate_ca(**kwargs)
            logger.info(f"{notify_prefix()} CA activated")
        except Exception as e:
            logger.warning(f"{notify_prefix()} CA activate skipped/failed: {e}")

    def close(self) -> None:
        if self.api is None:
            return
        try:
            self.api.logout()
        except Exception:
            pass
        self.api = None

    def _contract(self):
        assert self.api is not None
        # Prefer Contracts.Futures.TMF[TMFR1]; fall back to contracts.get
        try:
            return self.api.Contracts.Futures.TMF[self.contract_code]
        except Exception:
            return self.api.Contracts.Futures.TMF[self.instrument]

    def _reject(self, action: TradeAction, reason: str, why: str) -> None:
        self.rejects.append({"action": action.value, "reason": reason, "why": why})
        logger.warning(f"{notify_prefix()} sinopac reject {action.value}: {why} | {reason}")

    def place_tmf(
        self,
        action: TradeAction | str,
        price: float,
        *,
        reason: str = "",
        quantity: int = 1,
        dry_run: bool = False,
    ) -> bool:
        """
        Place one TMF futures order on SIM (or LIVE if allow_live).

        action: BUY / SELL open, or CLOSE (cover via opposite Auto).
        quantity forced to 1; open blocked if already in position.
        """
        if isinstance(action, str):
            action = TradeAction(action.upper())
        qty = int(quantity)
        if qty != 1:
            self._reject(action, reason, "quantity_must_be_1")
            return False

        if action in (TradeAction.BUY, TradeAction.SELL):
            if self.lots >= self.max_lots or self.position != Side.FLAT:
                self._reject(action, reason, "max_lots_or_already_in")
                return False
            side_open = Side.LONG if action == TradeAction.BUY else Side.SHORT
            sj_action = "Buy" if action == TradeAction.BUY else "Sell"
            oc = "New"
        elif action == TradeAction.CLOSE:
            if self.lots <= 0 or self.position == Side.FLAT:
                self._reject(action, reason, "flat")
                return False
            # close long → Sell; close short → Buy
            sj_action = "Sell" if self.position == Side.LONG else "Buy"
            oc = "Cover"
            side_open = self.position
        else:
            return False

        if dry_run:
            rec = {
                "action": action.value,
                "sj_action": sj_action,
                "octype": oc,
                "price": float(price),
                "quantity": 1,
                "dry_run": True,
                "reason": reason,
            }
            self.fills.append(rec)
            logger.info(f"{notify_prefix()} sinopac DRY {action.value} @ {price} | {reason}")
            # update local state for dry-run path so callers can exercise max_lots
            if action in (TradeAction.BUY, TradeAction.SELL):
                self.position = side_open
                self.lots = 1
            else:
                self.position = Side.FLAT
                self.lots = 0
            return True

        if not self.connect():
            self._reject(action, reason, "not_connected")
            return False

        try:
            import shioaji as sj

            contract = self._contract()
            action_enum = sj.Action.Buy if sj_action == "Buy" else sj.Action.Sell
            # API shape differs slightly across shioaji versions
            FuturesOrder = getattr(sj, "FuturesOrder", None) or getattr(
                sj.order, "FuturesOrder", None
            )
            if FuturesOrder is None:
                raise RuntimeError("FuturesOrder not found in shioaji")
            price_type = getattr(
                getattr(sj, "constant", sj), "FuturesPriceType", None
            )
            order_type = getattr(getattr(sj, "constant", sj), "OrderType", None)
            oc_type = getattr(getattr(sj, "constant", sj), "FuturesOCType", None)
            kwargs: dict[str, Any] = {
                "action": action_enum,
                "price": float(price),
                "quantity": 1,
                "account": self.api.futopt_account,
            }
            if price_type is not None:
                kwargs["price_type"] = price_type.LMT
            else:
                kwargs["price_type"] = "LMT"
            if order_type is not None:
                kwargs["order_type"] = order_type.ROD
            else:
                kwargs["order_type"] = "ROD"
            if oc_type is not None:
                kwargs["octype"] = oc_type.New if oc == "New" else oc_type.Cover
            else:
                kwargs["octype"] = oc
            order = FuturesOrder(**kwargs)
            trade = self.api.place_order(contract, order)
            try:
                self.api.update_status(trade)
            except Exception:
                pass
            self.last_trade = trade
            status = getattr(getattr(trade, "status", None), "status", None)
            status_s = str(status) if status is not None else ""
            ok = "Fail" not in status_s and "Cancel" not in status_s
            rec = {
                "action": action.value,
                "sj_action": sj_action,
                "octype": oc,
                "price": float(price),
                "quantity": 1,
                "status": status_s,
                "reason": reason,
            }
            if not ok:
                self.rejects.append({**rec, "why": "broker_status"})
                logger.warning(f"{notify_prefix()} sinopac order not ok: {status_s}")
                return False
            self.fills.append(rec)
            if action in (TradeAction.BUY, TradeAction.SELL):
                self.position = side_open
                self.lots = 1
            else:
                self.position = Side.FLAT
                self.lots = 0
            logger.info(
                f"{notify_prefix()} sinopac {action.value} @ {price} status={status_s} | {reason}"
            )
            return True
        except Exception as e:
            self._reject(action, reason, f"place_failed:{e}")
            return False

    def summary(self) -> dict[str, Any]:
        return {
            "simulation": self.simulation,
            "connected": self.api is not None,
            "position": self.position.value,
            "lots": self.lots,
            "max_lots": self.max_lots,
            "n_fills": len(self.fills),
            "n_rejects": len(self.rejects),
            "contract_code": self.contract_code,
        }
