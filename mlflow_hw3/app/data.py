import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

COLUMN_ALIASES = {
    "tickvol": "tick_volume",
    "tickvolume": "tick_volume",
    "realvolume": "real_volume",
}
REQUIRED_COLUMNS = ["open", "high", "low", "close"]
TIME_COLUMNS = ("datetime", "date", "timestamp")


def resolve_data_path() -> Path:
    configured = os.getenv("DATA_PATH")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            Path("data/brent_prices.csv"),
            Path("data/BRENT_D1.csv"),
            Path("../input/data/BRENT_H1.csv"),
            Path("../input/clean_df/BRENT_H1_clean.csv"),
            Path("input/data/BRENT_H1.csv"),
            Path("input/clean_df/BRENT_H1_clean.csv"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Dataset not found. Checked: {checked}")


def read_dataset(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="cp1251")
    if frame.shape[1] == 1:
        frame = pd.read_csv(path, sep=";")
    return frame


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = [re.sub(r"\s+", "_", str(column).strip().lower()) for column in frame.columns]
    frame = frame.rename(
        columns={column: COLUMN_ALIASES.get(column, column) for column in frame.columns}
    )
    if "time" not in frame:
        for candidate in TIME_COLUMNS:
            if candidate in frame:
                frame = frame.rename(columns={candidate: "time"})
                break
    if "tick_volume" not in frame:
        for candidate in ("volume", "real_volume"):
            if candidate in frame:
                frame["tick_volume"] = frame[candidate]
                break
    if "tick_volume" not in frame:
        frame["tick_volume"] = 0.0
    return frame


def normalize_candles(data: pd.DataFrame) -> pd.DataFrame:
    frame = standardize_columns(data)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    numeric_columns = [*REQUIRED_COLUMNS, "tick_volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    frame = frame[frame[REQUIRED_COLUMNS].gt(0).all(axis=1)]
    frame = frame[frame["high"] >= frame[["open", "close", "low"]].max(axis=1)]
    frame = frame[frame["low"] <= frame[["open", "close", "high"]].min(axis=1)]
    frame["tick_volume"] = frame["tick_volume"].fillna(0).clip(lower=0)

    if "time" in frame:
        parsed_time = pd.to_datetime(frame["time"], errors="coerce", utc=True)
        if not parsed_time.notna().any():
            raise ValueError("Dataset contains a time column but no valid timestamps")
        frame["time"] = parsed_time
        frame = frame[frame["time"].notna()]
        frame = frame.sort_values("time", kind="stable")
        frame = frame.drop_duplicates(subset=["time"], keep="last")

    return frame.reset_index(drop=True)


def load_candles(path: Path) -> pd.DataFrame:
    return normalize_candles(read_dataset(path))


def data_quality_report(source: Path, candles: pd.DataFrame) -> dict[str, Any]:
    raw = standardize_columns(read_dataset(source))
    report: dict[str, Any] = {
        "source_path": str(source.resolve()),
        "source_rows": int(len(raw)),
        "rows_after_cleaning": int(len(candles)),
        "rows_removed": int(len(raw) - len(candles)),
        "missing_values_by_column": {
            str(column): int(count) for column, count in raw.isna().sum().items()
        },
    }

    if "time" in raw:
        timestamps = pd.to_datetime(raw["time"], errors="coerce", utc=True)
        report["invalid_timestamps"] = int(timestamps.isna().sum())
        report["duplicate_timestamps"] = int(timestamps.duplicated().sum())

    ohlc = raw[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    report["flat_ohlc_rows"] = int(ohlc.nunique(axis=1).eq(1).sum())
    report["flat_ohlc_share"] = float(ohlc.nunique(axis=1).eq(1).mean())
    volume = pd.to_numeric(raw["tick_volume"], errors="coerce").fillna(0)
    report["zero_volume_rows"] = int(volume.eq(0).sum())
    report["zero_volume_share"] = float(volume.eq(0).mean())
    report["first_timestamp"] = str(candles["time"].iloc[0]) if "time" in candles else None
    report["last_timestamp"] = str(candles["time"].iloc[-1]) if "time" in candles else None
    return report
