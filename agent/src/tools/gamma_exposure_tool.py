"""Gamma exposure (GEX) tool: aggregate net dealer gamma exposure by strike
from a US-listed options chain, via the shared Yahoo Finance client.

Gold CFD/forex (XAU/USD) trades over-the-counter with no centralized options
book, so there is no direct gamma-exposure feed for it. GLD (the SPDR Gold
Shares ETF) is the closest exchange-listed, options-liquid proxy and is the
default underlying here. GLD's share price tracks a fraction of spot gold
that drifts slowly (the trust's expense ratio erodes the backing over time),
so strikes should be scaled to an XAU/USD-equivalent level via the live ratio
(current XAUUSD spot / this tool's reported spot), not a hardcoded constant.

The GEX figure itself is a retail approximation, not verified dealer
positioning (real market-maker inventory is proprietary data): it follows the
common convention used by most public gamma-exposure calculators, treating
dealer gamma as long on calls and short on puts.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

from backtest.loaders import yahoo_client
from src.agent.tools import BaseTool

_CONTRACT_MULTIPLIER = 100  # shares per US equity/ETF option contract
_DEFAULT_NUM_EXPIRATIONS = 12
_MAX_NUM_EXPIRATIONS = 15
_TOP_STRIKES_LIMIT = 8
# Yahoo's free/delayed feed fills illiquid contracts with placeholder implied
# vol (commonly exactly 0.00001, or implausibly >300%) instead of omitting
# the field; both are noise, not a real market quote, so they're excluded.
_MIN_PLAUSIBLE_IV = 0.01
_MAX_PLAUSIBLE_IV = 3.0


class GammaExposureTool(BaseTool):
    """Aggregate net gamma exposure (GEX) by strike for a US-listed underlying."""

    name = "get_gamma_exposure"
    description = (
        "Compute an approximate dealer gamma exposure (GEX) profile by strike "
        "from a US-listed options chain (Yahoo Finance), aggregated across the "
        "nearest expirations. Defaults to GLD (the gold ETF) as a proxy for "
        "XAU/USD, since gold CFD/forex has no centralized options book. Returns "
        "net GEX by strike, the estimated zero-gamma ('flip') level, the "
        "largest gamma strikes (support/resistance-like zones), and the "
        "overall positive/negative gamma regime (positive = range-dampening, "
        "negative = volatility-amplifying). This is a retail approximation "
        "(assumes dealers long calls / short puts), not verified market-maker "
        "positioning. Example: get_gamma_exposure() for the default GLD proxy."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": (
                    "US-listed underlying with a liquid options chain. "
                    "Default 'GLD' (gold ETF proxy for XAU/USD)."
                ),
                "default": "GLD",
            },
            "num_expirations": {
                "type": "integer",
                "description": (
                    "How many of the nearest expirations to aggregate "
                    "(default 12, max 15). GLD's near-daily/weekly expirations "
                    "often carry ~zero open interest, so a low value can "
                    "return little or no data (or calls-only); raise it toward "
                    "15 if the result looks thin or one-sided."
                ),
                "default": _DEFAULT_NUM_EXPIRATIONS,
            },
            "risk_free_rate": {
                "type": "number",
                "description": "Annualized risk-free rate used in the Black-Scholes gamma calc.",
                "default": 0.05,
            },
        },
        "required": [],
    }

    def execute(self, **kwargs: Any) -> str:
        """Fetch the options chain and return a JSON-string GEX-by-strike envelope.

        Args:
            **kwargs: Optional ``ticker`` (default "GLD"), ``num_expirations``
                (default 4, capped at 8), ``risk_free_rate`` (default 0.05).

        Returns:
            A JSON string. On success:
            ``{"ok": true, "source": "yahoo", "data": {...}}`` with ``spot``,
            ``total_net_gex``, ``regime``, ``gamma_flip_strike``,
            ``top_gamma_strikes``, and the full ``by_strike`` breakdown. On
            failure: ``{"ok": false, "error": str}``.
        """
        ticker = str(kwargs.get("ticker") or "GLD").strip().upper()
        risk_free_rate = float(kwargs.get("risk_free_rate", 0.05))
        num_expirations = _coerce_num_expirations(kwargs.get("num_expirations"))

        try:
            first = yahoo_client.get_options(ticker)
        except Exception as exc:  # noqa: BLE001 - surface as error envelope
            return _error(f"yahoo options request failed: {exc}")

        spot = _extract_spot(first)
        if spot is None or spot <= 0:
            return _error(f"no live quote price available for {ticker}")

        expirations = [e for e in (first.get("expirationDates") or []) if e is not None]
        if not expirations:
            return _error(f"no options expirations available for {ticker}")

        selected_expirations = expirations[:num_expirations]

        now = time.time()
        strikes: Dict[float, Dict[str, float]] = {}
        expirations_used: List[int] = []

        for idx, expiration in enumerate(selected_expirations):
            if idx == 0:
                block = _first_options_block(first)
            else:
                try:
                    chain = yahoo_client.get_options(ticker, expiration=expiration)
                except Exception:  # noqa: BLE001 - skip a bad expiration, keep going
                    continue
                block = _first_options_block(chain)
            if not block:
                continue
            expirations_used.append(expiration)
            T = max((expiration - now) / (365.0 * 86400.0), 0.0)
            _accumulate_gex(strikes, block.get("calls"), "call", spot, T, risk_free_rate)
            _accumulate_gex(strikes, block.get("puts"), "put", spot, T, risk_free_rate)

        if not strikes:
            return _error(f"no contracts with usable implied volatility for {ticker}")

        rows = _build_strike_rows(strikes)
        total_net_gex = round(sum(row["net_gex"] for row in rows), 2)
        top_strikes = sorted(rows, key=lambda row: abs(row["net_gex"]), reverse=True)[:_TOP_STRIKES_LIMIT]

        data = {
            "ticker": ticker,
            "spot": spot,
            "expirations_used": expirations_used,
            "num_strikes": len(rows),
            "total_net_gex": total_net_gex,
            "regime": "positive" if total_net_gex >= 0 else "negative",
            "gamma_flip_strike": _find_flip_strike(rows),
            "top_gamma_strikes": top_strikes,
            "by_strike": rows,
            "note": (
                "Approximate GEX (assumes dealers long calls / short puts); "
                f"{ticker} is a proxy underlying, not the traded CFD instrument. "
                "Scale strikes to an XAU/USD-equivalent level via "
                "(current XAUUSD spot / this spot) before comparing to the gold chart."
            ),
        }
        return json.dumps({"ok": True, "source": "yahoo", "data": data}, ensure_ascii=False)


def _coerce_num_expirations(value: Any) -> int:
    """Coerce and clamp the requested expiration count to [1, _MAX_NUM_EXPIRATIONS]."""
    try:
        n = int(value) if value is not None else _DEFAULT_NUM_EXPIRATIONS
    except (TypeError, ValueError):
        n = _DEFAULT_NUM_EXPIRATIONS
    return max(1, min(n, _MAX_NUM_EXPIRATIONS))


def _extract_spot(result: Dict[str, Any]) -> Optional[float]:
    """Pull the live underlying price from the options-chain quote block."""
    quote = result.get("quote") or {}
    price = quote.get("regularMarketPrice")
    try:
        return float(price) if price is not None else None
    except (TypeError, ValueError):
        return None


def _first_options_block(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the single expiration's calls/puts block from a chain result."""
    options = result.get("options") or []
    return options[0] if options else {}


def _accumulate_gex(
    strikes: Dict[float, Dict[str, float]],
    contracts: Any,
    side: str,
    spot: float,
    T: float,
    risk_free_rate: float,
) -> None:
    """Fold one side's (calls/puts) contracts for one expiration into ``strikes``."""
    if not isinstance(contracts, list):
        return
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        try:
            strike = float(contract.get("strike"))
            oi = float(contract.get("openInterest") or 0)
            iv_raw = contract.get("impliedVolatility")
            iv = float(iv_raw) if iv_raw is not None else None
        except (TypeError, ValueError):
            continue
        if strike <= 0 or oi <= 0 or not iv:
            continue
        if iv < _MIN_PLAUSIBLE_IV or iv > _MAX_PLAUSIBLE_IV:
            continue

        gamma = _bs_gamma(spot, strike, T, risk_free_rate, iv)
        gex = oi * gamma * spot * spot * 0.01 * _CONTRACT_MULTIPLIER

        row = strikes.setdefault(
            strike, {"call_gex": 0.0, "put_gex": 0.0, "call_oi": 0.0, "put_oi": 0.0}
        )
        if side == "call":
            row["call_gex"] += gex
            row["call_oi"] += oi
        else:
            row["put_gex"] += gex
            row["put_oi"] += oi


def _bs_gamma(spot: float, strike: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (identical for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(spot / strike) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    return float(norm.pdf(d1) / (spot * sigma * sqrt_T))


def _build_strike_rows(strikes: Dict[float, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Sort strikes ascending and compute each row's net and cumulative GEX."""
    rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    for strike in sorted(strikes.keys()):
        row = strikes[strike]
        net_gex = row["call_gex"] - row["put_gex"]
        cumulative += net_gex
        rows.append(
            {
                "strike": strike,
                "call_open_interest": row["call_oi"],
                "put_open_interest": row["put_oi"],
                "net_gex": round(net_gex, 2),
                "cumulative_gex": round(cumulative, 2),
            }
        )
    return rows


def _find_flip_strike(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Find the strike where cumulative GEX (ascending) first crosses zero."""
    prev_cumulative = 0.0
    for row in rows:
        cumulative = row["cumulative_gex"]
        if (prev_cumulative < 0 <= cumulative) or (prev_cumulative > 0 >= cumulative):
            return row["strike"]
        prev_cumulative = cumulative
    return None


def _error(message: str) -> str:
    """Render a failure envelope as a JSON string."""
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)
