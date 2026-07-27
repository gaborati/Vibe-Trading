"""Options open-interest heatmap tool: strike x expiration OI grid via Yahoo.

Complements get_gamma_exposure with the raw open-interest picture behind it
(no Black-Scholes/implied-vol filtering, so it surfaces every liquid strike
even where Yahoo's implied volatility field is missing or implausible).
Defaults to GLD (the gold ETF) as a proxy for XAU/USD, for the same reason as
the gamma exposure tool: gold CFD/forex has no centralized options book.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backtest.loaders import yahoo_client
from src.agent.tools import BaseTool

_DEFAULT_NUM_EXPIRATIONS = 12
_MAX_NUM_EXPIRATIONS = 15
_MAX_GRID_CELLS = 60
_MAX_WALLS = 10


class OptionsOIHeatmapTool(BaseTool):
    """Aggregate call/put open interest into a strike x expiration heatmap grid."""

    name = "get_options_oi_heatmap"
    description = (
        "Build an open-interest heatmap (strike x expiration) from a "
        "US-listed options chain (Yahoo Finance), across the nearest "
        "expirations. Defaults to GLD (gold ETF) as a proxy for XAU/USD, "
        "since gold CFD/forex has no centralized options book. Returns the "
        "highest-OI cells (strike/expiration pairs), the overall top OI "
        "'walls' aggregated across expirations, and each expiration's single "
        "biggest strike (term structure). Complements get_gamma_exposure with "
        "the raw OI picture (no implied-vol filtering, so it catches more "
        "strikes). Example: get_options_oi_heatmap() for the default GLD proxy."
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
                    "How many of the nearest expirations to include "
                    "(default 12, max 15)."
                ),
                "default": _DEFAULT_NUM_EXPIRATIONS,
            },
        },
        "required": [],
    }

    def execute(self, **kwargs: Any) -> str:
        """Fetch the options chain and return a JSON-string OI-heatmap envelope.

        Args:
            **kwargs: Optional ``ticker`` (default "GLD") and
                ``num_expirations`` (default 12, capped at 15).

        Returns:
            A JSON string. On success:
            ``{"ok": true, "source": "yahoo", "data": {...}}`` with ``spot``,
            ``grid`` (top OI cells), ``top_walls`` (aggregated across
            expirations), and ``by_expiration_top_strike`` (term structure).
            On failure: ``{"ok": false, "error": str}``.
        """
        ticker = str(kwargs.get("ticker") or "GLD").strip().upper()
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

        cells: List[Dict[str, Any]] = []
        by_expiration_top: List[Dict[str, Any]] = []
        totals: Dict[float, Dict[str, float]] = {}
        expirations_used: List[Dict[str, Any]] = []

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

            expirations_used.append(
                {"expiration": expiration, "date": _to_date_string(expiration)}
            )

            per_strike = _accumulate_oi(block)
            best_strike = None
            best_oi = 0.0
            for strike, (call_oi, put_oi) in per_strike.items():
                total_oi = call_oi + put_oi
                if total_oi <= 0:
                    continue
                cells.append(
                    {
                        "expiration": expiration,
                        "date": _to_date_string(expiration),
                        "strike": strike,
                        "call_oi": call_oi,
                        "put_oi": put_oi,
                        "total_oi": total_oi,
                    }
                )
                row = totals.setdefault(strike, {"call_oi": 0.0, "put_oi": 0.0})
                row["call_oi"] += call_oi
                row["put_oi"] += put_oi
                if total_oi > best_oi:
                    best_oi = total_oi
                    best_strike = strike
            if best_strike is not None:
                by_expiration_top.append(
                    {
                        "expiration": expiration,
                        "date": _to_date_string(expiration),
                        "top_strike": best_strike,
                        "top_oi": best_oi,
                    }
                )

        if not cells:
            return _error(f"no open-interest data available for {ticker}")

        cells.sort(key=lambda c: c["total_oi"], reverse=True)
        top_walls = sorted(
            (
                {
                    "strike": strike,
                    "call_oi": row["call_oi"],
                    "put_oi": row["put_oi"],
                    "total_oi": row["call_oi"] + row["put_oi"],
                }
                for strike, row in totals.items()
            ),
            key=lambda r: r["total_oi"],
            reverse=True,
        )[:_MAX_WALLS]

        data = {
            "ticker": ticker,
            "spot": spot,
            "expirations_used": expirations_used,
            "grid": cells[:_MAX_GRID_CELLS],
            "top_walls": top_walls,
            "by_expiration_top_strike": by_expiration_top,
            "note": (
                f"{ticker} is a proxy underlying, not the traded CFD instrument. "
                "Scale strikes to an XAU/USD-equivalent level via "
                "(current XAUUSD spot / this spot) before comparing to the gold "
                "chart. 'grid' is capped to the highest-OI cells; 'top_walls' is "
                "aggregated across all requested expirations."
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


def _accumulate_oi(block: Dict[str, Any]) -> Dict[float, tuple[float, float]]:
    """Sum call/put open interest per strike for one expiration's chain block."""
    per_strike: Dict[float, List[float]] = {}
    for side, contracts in (("call", block.get("calls")), ("put", block.get("puts"))):
        if not isinstance(contracts, list):
            continue
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            try:
                strike = float(contract.get("strike"))
                oi = float(contract.get("openInterest") or 0)
            except (TypeError, ValueError):
                continue
            if strike <= 0 or oi <= 0:
                continue
            row = per_strike.setdefault(strike, [0.0, 0.0])
            row[0 if side == "call" else 1] += oi
    return {strike: (row[0], row[1]) for strike, row in per_strike.items()}


def _to_date_string(epoch_seconds: int) -> str:
    """Render an epoch-second expiration as an ISO date string (UTC)."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%d")


def _error(message: str) -> str:
    """Render a failure envelope as a JSON string."""
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)
