"""
Tradovate REST client — Python twin of mission-control's lib/tradovate.ts,
sharing the same credentials (TRADOVATE_* env vars, falling back to
mission-control's .env.local) and the same conventions: token cache with
renew-before-expiry, p-ticket rate-limit surfacing, isAutomated on orders.

HARD DEMO RAIL: order-placing methods refuse unless the resolved environment
is "demo". Going live is a deliberate code change (flip ALLOW_LIVE below),
not a config accident — mirroring the paranoia of mission-control's
/api/broker/order route.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

ALLOW_LIVE = False           # deliberate code-change gate, not a config flag
_KEY_FILE = r"C:\Users\angel\claude code\mission-control\.env.local"
_ENV_KEYS = ("TRADOVATE_NAME", "TRADOVATE_PASSWORD", "TRADOVATE_CID",
             "TRADOVATE_SECRET", "TRADOVATE_APP_ID", "TRADOVATE_ENV")


def _load_env() -> dict:
    vals = {k: os.environ.get(k) for k in _ENV_KEYS}
    if not all(vals[k] for k in _ENV_KEYS[:4]):     # fall back to the file
        try:
            with open(_KEY_FILE) as f:
                for line in f:
                    m = re.match(r"\s*(TRADOVATE_[A-Z_]+)\s*=\s*['\"]?([^'\"\n]+)",
                                 line)
                    if m and not vals.get(m.group(1)):
                        vals[m.group(1)] = m.group(2).strip()
        except OSError:
            pass
    return vals


@dataclass
class TvConfig:
    env: str
    name: str
    password: str
    cid: int
    secret: str
    app_id: str

    @classmethod
    def load(cls) -> "TvConfig":
        v = _load_env()
        missing = [k for k in _ENV_KEYS[:4] if not v.get(k)]
        if missing:
            raise RuntimeError(f"Tradovate credentials missing: {missing}")
        return cls(
            env="live" if v.get("TRADOVATE_ENV") == "live" else "demo",
            name=v["TRADOVATE_NAME"], password=v["TRADOVATE_PASSWORD"],
            cid=int(v["TRADOVATE_CID"]), secret=v["TRADOVATE_SECRET"],
            app_id=v.get("TRADOVATE_APP_ID") or "QuantReal",
        )


class TradovateClient:
    def __init__(self, cfg: Optional[TvConfig] = None):
        self.cfg = cfg or TvConfig.load()
        self._token: Optional[str] = None
        self._expires_at = 0.0

    @property
    def base(self) -> str:
        return f"https://{self.cfg.env}.tradovateapi.com/v1"

    # -- auth ----------------------------------------------------------------

    def _authenticate(self) -> str:
        now = time.time()
        if self._token and self._expires_at - now > 600:
            return self._token
        if self._token and self._expires_at > now:      # renew
            try:
                r = requests.get(f"{self.base}/auth/renewaccesstoken",
                                 headers={"Authorization": f"Bearer {self._token}"},
                                 timeout=20)
                if r.ok and r.json().get("accessToken"):
                    d = r.json()
                    self._token = d["accessToken"]
                    self._expires_at = _parse_exp(d.get("expirationTime"))
                    return self._token
            except requests.RequestException:
                pass                                     # full auth below

        r = requests.post(f"{self.base}/auth/accesstokenrequest", json={
            "name": self.cfg.name, "password": self.cfg.password,
            "appId": self.cfg.app_id, "appVersion": "1.0.0",
            "cid": self.cfg.cid, "sec": self.cfg.secret,
            "deviceId": f"quant-real-{socket.gethostname()}"[:64],
        }, timeout=30)
        d = r.json() if r.content else {}
        if d.get("p-ticket"):
            raise RuntimeError(
                f"Tradovate is rate-limiting auth — wait {d.get('p-time', '?')}s")
        if not r.ok or d.get("errorText") or not d.get("accessToken"):
            raise RuntimeError(d.get("errorText")
                               or f"Tradovate auth failed ({r.status_code})")
        self._token = d["accessToken"]
        self._expires_at = _parse_exp(d.get("expirationTime"))
        logger.info("Tradovate auth OK (%s)", self.cfg.env)
        return self._token

    def _req(self, method: str, endpoint: str, body: Any = None) -> Any:
        token = self._authenticate()
        r = requests.request(method, f"{self.base}{endpoint}",
                             headers={"Authorization": f"Bearer {token}"},
                             json=body, timeout=30)
        if r.status_code == 401:
            self._token = None
            raise RuntimeError("Tradovate session expired — retry")
        d = r.json() if r.content else {}
        if not r.ok:
            raise RuntimeError((d or {}).get("errorText")
                               or f"Tradovate {endpoint} failed ({r.status_code})")
        return d

    def get(self, endpoint: str) -> Any:
        return self._req("GET", endpoint)

    def post(self, endpoint: str, body: Any) -> Any:
        return self._req("POST", endpoint, body)

    # -- reads ---------------------------------------------------------------

    def account(self) -> dict:
        accounts = self.get("/account/list")
        act = next((a for a in accounts if a.get("active")), None) or accounts[0]
        return act

    def positions(self) -> list[dict]:
        return [p for p in self.get("/position/list") if p.get("netPos")]

    def contract_find(self, symbol: str) -> dict:
        return self.get(f"/contract/find?name={symbol}")

    def cash_snapshot(self, account_id: int) -> dict:
        items = self.get("/cashBalance/list")
        return next((c for c in items if c.get("accountId") == account_id), {})

    # -- orders (demo-railed) --------------------------------------------------

    def _guard(self):
        if self.cfg.env != "demo" and not ALLOW_LIVE:
            raise RuntimeError(
                "Refusing to place orders: TRADOVATE_ENV is not 'demo' and "
                "ALLOW_LIVE is False (edit execution/tradovate_client.py to "
                "go live — deliberately not a config flag).")

    def place_order(self, *, action: str, symbol: str, qty: int,
                    order_type: str = "Market",
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> dict:
        self._guard()
        if action not in ("Buy", "Sell"):
            raise ValueError("action must be Buy or Sell")
        if not (1 <= qty <= 20):
            raise ValueError("qty must be 1-20 (demo rail)")
        acct = self.account()
        body = {"accountId": acct["id"], "accountSpec": acct["name"],
                "action": action, "symbol": symbol, "orderQty": qty,
                "orderType": order_type, "isAutomated": True}
        if order_type == "Limit":
            body["price"] = price
        if order_type == "Stop":
            body["stopPrice"] = stop_price
        result = self.post("/order/placeorder", body)
        if result.get("failureReason") or result.get("errorText"):
            raise RuntimeError(result.get("failureText")
                               or result.get("errorText")
                               or result["failureReason"])
        logger.info("order placed: %s %d %s (%s) -> id %s",
                    action, qty, symbol, order_type, result.get("orderId"))
        return result

    def liquidate(self, contract_id: int) -> dict:
        self._guard()
        acct = self.account()
        return self.post("/order/liquidateposition",
                         {"accountId": acct["id"], "contractId": contract_id,
                          "admin": False})


def _parse_exp(iso: Optional[str]) -> float:
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return time.time() + 60 * 60
