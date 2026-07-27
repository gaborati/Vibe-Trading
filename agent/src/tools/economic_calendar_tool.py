"""Read-only tool: upcoming/recent US macro release dates (FRED release calendar).

Gold is highly sensitive to a handful of high-impact US macro releases (CPI,
the jobs report, GDP, the Fed's preferred inflation gauge). FRED publishes the
official release calendar for each of these well in advance via its
``fred/release/dates`` endpoint, keyed by a release ID. This tool queries a
small, curated set of gold-relevant release IDs and merges them into one
sorted calendar so the agent can flag "a high-impact release lands inside
this trade's holding window" before proposing an entry.

Like get_macro_series, this reads the free FRED API key from FRED_API_KEY and
is silently excluded from the registry when that key is absent.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

from backtest.loaders._http import resolve_min_interval, throttled_get_json
from src.agent.tools import BaseTool
from src.config.accessor import get_env_config

logger = logging.getLogger(__name__)

_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"

_FRED_HOST_KEY = "fred"
_FRED_MIN_INTERVAL_ENV = "VIBE_TRADING_FRED_MIN_INTERVAL"
_FRED_DEFAULT_MIN_INTERVAL = 0.6
_FRED_TIMEOUT_S = 15.0

# Curated, gold-relevant US macro releases -- the handful that reliably move
# XAU/USD via their effect on rate expectations and the dollar. release_id
# values confirmed against fred.stlouisfed.org/release?rid=<id>.
_GOLD_RELEVANT_RELEASES: dict[str, str] = {
    "50": "Employment Situation (NFP + unemployment rate)",
    "10": "Consumer Price Index (CPI)",
    "54": "Personal Income and Outlays (PCE, the Fed's preferred inflation gauge)",
    "53": "Gross Domestic Product (GDP)",
}

_DEFAULT_DAYS_BACK = 3
_DEFAULT_DAYS_AHEAD = 30
_MAX_DAYS_AHEAD = 180


class EconomicCalendarTool(BaseTool):
    """Fetch upcoming/recent dates for a curated set of gold-relevant US macro releases."""

    name = "get_economic_calendar"
    description = (
        "Fetch upcoming and recent release dates for the small set of US "
        "macro releases that most reliably move gold: the jobs report "
        "(Employment Situation / NFP), CPI, PCE (Personal Income and "
        "Outlays), and GDP -- sourced from FRED's official release calendar. "
        "Use this before proposing a trade to check whether a high-impact "
        "release falls inside the intended holding window (spread widening / "
        "whipsaw risk around the release time). Requires a free FRED API key "
        "(FRED_API_KEY, same one used by get_macro_series). Note: FRED gives "
        "the release DATE, not an exact intraday time or forecast/consensus "
        "value -- most of these US releases land around 8:30am America/"
        "New_York on their scheduled date, but confirm the exact time "
        "separately if it matters for a same-day trade."
    )
    parameters = {
        "type": "object",
        "properties": {
            "days_back": {
                "type": "integer",
                "description": "How many days before today to include (default 3).",
                "default": _DEFAULT_DAYS_BACK,
            },
            "days_ahead": {
                "type": "integer",
                "description": (
                    f"How many days after today to include (default "
                    f"{_DEFAULT_DAYS_AHEAD}, max {_MAX_DAYS_AHEAD})."
                ),
                "default": _DEFAULT_DAYS_AHEAD,
            },
        },
        "required": [],
    }

    @classmethod
    def check_available(cls) -> bool:
        """Available only when a FRED API key is configured.

        Returns:
            ``True`` when ``FRED_API_KEY`` is set in the environment, otherwise
            ``False`` so the tool is silently excluded from the registry.
        """
        return bool(get_env_config().data.fred_api_key)

    def execute(self, **kwargs: Any) -> str:
        """Fetch and merge release dates across the curated release set.

        Args:
            **kwargs: Optional ``days_back`` (default 3) and ``days_ahead``
                (default 30, capped at 180).

        Returns:
            A JSON string envelope. On success:
            ``{"ok": true, "source": "fred", "data": {"today", "events": [
            {"date", "release_id", "release_name", "is_future"}, ...],
            "count"}}``, sorted ascending by date. On failure:
            ``{"ok": false, "error": str}``.
        """
        api_key = get_env_config().data.fred_api_key or None
        if not api_key:
            return _error("FRED_API_KEY is not configured")

        today = dt.date.today()
        days_back = _coerce_days(kwargs.get("days_back"), _DEFAULT_DAYS_BACK, max_value=365)
        days_ahead = _coerce_days(
            kwargs.get("days_ahead"), _DEFAULT_DAYS_AHEAD, max_value=_MAX_DAYS_AHEAD
        )
        window_start = today - dt.timedelta(days=days_back)
        window_end = today + dt.timedelta(days=days_ahead)

        events: list[dict[str, Any]] = []
        errors: list[str] = []
        for release_id, release_name in _GOLD_RELEVANT_RELEASES.items():
            try:
                dates = _fetch_release_dates(
                    release_id=release_id,
                    api_key=api_key,
                    start=window_start,
                    end=window_end,
                )
            except Exception as exc:  # noqa: BLE001 - one bad release shouldn't drop the rest
                errors.append(f"release {release_id} ({release_name}): {exc}")
                continue
            for date_str in dates:
                events.append(
                    {
                        "date": date_str,
                        "release_id": release_id,
                        "release_name": release_name,
                        "is_future": date_str >= today.isoformat(),
                    }
                )

        if not events and errors:
            return _error("; ".join(errors))

        events.sort(key=lambda e: e["date"])

        data: dict[str, Any] = {
            "today": today.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "events": events,
            "count": len(events),
        }
        if errors:
            data["partial_errors"] = errors

        return json.dumps({"ok": True, "source": "fred", "data": data}, ensure_ascii=False)


def _fetch_release_dates(
    *, release_id: str, api_key: str, start: dt.date, end: dt.date
) -> list[str]:
    """Fetch one release's scheduled dates within [start, end] from FRED.

    ``include_release_dates_with_no_data=true`` is required to surface future
    dates FRED has scheduled but not yet published data for -- without it,
    genuinely upcoming releases are silently dropped.
    """
    params = {
        "release_id": release_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "realtime_start": start.isoformat(),
        "realtime_end": end.isoformat(),
        "include_release_dates_with_no_data": "true",
    }
    payload = throttled_get_json(
        _RELEASE_DATES_URL,
        host_key=_FRED_HOST_KEY,
        min_interval=resolve_min_interval(_FRED_MIN_INTERVAL_ENV, _FRED_DEFAULT_MIN_INTERVAL),
        params=params,
        timeout=_FRED_TIMEOUT_S,
    )
    if not isinstance(payload, dict):
        return []
    rows = payload.get("release_dates")
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("date"), str) and row["date"].strip():
            out.append(row["date"].strip())
    return out


def _coerce_days(value: Any, default: int, *, max_value: int) -> int:
    """Coerce and clamp a requested day-count into [0, max_value]."""
    try:
        n = int(value) if value is not None else default
    except (TypeError, ValueError):
        n = default
    return max(0, min(n, max_value))


def _error(message: str) -> str:
    """Render a failure envelope as a JSON string."""
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)
