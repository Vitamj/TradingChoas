from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_10",
    "volatility_5",
    "volatility_10",
    "sma_gap_5",
    "sma_gap_10",
]
MIN_HISTORY_ROWS = 11


def build_feature_frame(candles: Sequence[dict[str, Any]]) -> pd.DataFrame:
    if len(candles) < MIN_HISTORY_ROWS:
        raise ValueError(f"At least {MIN_HISTORY_ROWS} candles are required")
    frame = pd.DataFrame(candles)
    if "date" in frame and frame["date"].notna().all():
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.sort_values("date", kind="stable")
        if frame["date"].duplicated().any():
            raise ValueError("Candle dates must be unique")
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    if close.isna().any() or (close <= 0).any():
        raise ValueError("Close prices must be positive finite numbers")
    returns = close.pct_change(fill_method=None)
    row: dict[str, float] = {}
    for period in (1, 2, 3, 5, 10):
        row[f"return_{period}"] = float(close.iloc[-1] / close.iloc[-period - 1] - 1)
    row["volatility_5"] = float(returns.rolling(5).std(ddof=0).iloc[-1])
    row["volatility_10"] = float(returns.rolling(10).std(ddof=0).iloc[-1])
    row["sma_gap_5"] = float(close.iloc[-1] / close.iloc[-5:].mean() - 1)
    row["sma_gap_10"] = float(close.iloc[-1] / close.iloc[-10:].mean() - 1)
    values = np.asarray(list(row.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The supplied history does not produce finite features")
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def candles_to_payload(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        date_value = row.get("date", row.get("time", row.get("datetime")))
        if isinstance(date_value, pd.Timestamp):
            date_value = date_value.isoformat()
        elif isinstance(date_value, datetime):
            date_value = date_value.isoformat()
        normalized.append(
            {
                "date": date_value,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", row.get("tick_volume", 0)) or 0),
            }
        )
    return normalized
