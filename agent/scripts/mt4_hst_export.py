"""Convert an MT4 ``.hst`` history file to the CSV format the ``local``
backtest loader expects (``date,open,high,low,close,volume``).

MT4 stores each chart's downloaded bars in a binary ``.hst`` file under
``MQL4/../history/<server>/<SYMBOL><PERIOD_MINUTES>.hst`` (e.g.
``XAUUSD60.hst`` for hourly gold bars). This lets the Vibe-Trading agent read
the exact broker feed you see in your own MT4 terminal instead of a proxy
data source, via the project's existing ``local`` data-bridge mechanism
(``~/.vibe-trading/data-bridge/config.yaml``).

Supports both on-disk record layouts MT4 has used:
  - version 400 (legacy): 44-byte records, 4-byte time, 4 float64 OHLC + volume
  - version 401 (current): 60-byte records, 8-byte time, OHLC, tick_volume,
    spread, real_volume

Usage:
    python mt4_hst_export.py <path-to-file.hst> <output.csv>
"""

from __future__ import annotations

import csv
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HEADER_SIZE = 148
_RECORD_SIZE_V400 = 44

# MT4 stores each bar's ctm using the broker server's own clock, not true
# UTC. Verified empirically on 2026-07-22 (CEST/UTC+2): rendering ctm with
# tz=utc produced values exactly 2 hours ahead of real UTC "now". Subtracting
# this offset before labeling the output "UTC" gives callers (including the
# LLM agent, which otherwise mis-converted and drifted by ~2-3 hours across
# a session) a value that is actually correct UTC, not just UTC-labeled
# broker time. This is a fixed offset, not DST-aware -- MetaQuotes-Demo's
# server clock may or may not shift with European DST; re-verify (compare a
# fresh export's last bar against `date -u`) if this drifts again, especially
# around DST transitions (late March / late October).
_BROKER_SERVER_UTC_OFFSET = timedelta(hours=2)
_RECORD_SIZE_V401 = 60


def _read_header(data: bytes) -> tuple[int, str, int]:
    version = struct.unpack_from("<i", data, 0)[0]
    symbol = data[68:80].split(b"\x00", 1)[0].decode("ascii", errors="replace")
    period = struct.unpack_from("<i", data, 80)[0]
    return version, symbol, period


def _iter_records_v400(body: bytes):
    count = len(body) // _RECORD_SIZE_V400
    for i in range(count):
        chunk = body[i * _RECORD_SIZE_V400 : (i + 1) * _RECORD_SIZE_V400]
        ctm, o, low, high, c, vol = struct.unpack("<i5d", chunk)
        yield ctm, o, high, low, c, vol


def _iter_records_v401(body: bytes):
    count = len(body) // _RECORD_SIZE_V401
    for i in range(count):
        chunk = body[i * _RECORD_SIZE_V401 : (i + 1) * _RECORD_SIZE_V401]
        ctm, o, h, low, c, tick_vol, _spread, real_vol = struct.unpack("<q4dqiq", chunk)
        vol = real_vol if real_vol else tick_vol
        yield ctm, o, h, low, c, vol


def convert(hst_path: Path, csv_path: Path) -> int:
    data = hst_path.read_bytes()
    if len(data) < _HEADER_SIZE:
        raise ValueError(f"{hst_path} is too small to be a valid .hst file")

    version, symbol, period = _read_header(data)
    body = data[_HEADER_SIZE:]

    if version == 401:
        records = list(_iter_records_v401(body))
    elif version == 400:
        records = list(_iter_records_v400(body))
    else:
        raise ValueError(f"unsupported .hst version {version} in {hst_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for ctm, o, h, low, c, vol in records:
            broker_time = datetime.fromtimestamp(ctm, tz=timezone.utc)
            true_utc = broker_time - _BROKER_SERVER_UTC_OFFSET
            ts = true_utc.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([ts, o, h, low, c, vol])

    print(f"{symbol} period={period}m version={version}: wrote {len(records)} bars -> {csv_path}")
    return len(records)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: mt4_hst_export.py <path-to-file.hst> <output.csv>", file=sys.stderr)
        raise SystemExit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
