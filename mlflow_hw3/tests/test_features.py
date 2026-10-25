from pathlib import Path

import pandas as pd

from app.data import load_candles, normalize_candles
from app.features import FEATURE_COLUMNS, MIN_HISTORY_ROWS, prepare_training_data


def make_candles(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "open": closes,
            "high": [value + 0.5 for value in closes],
            "low": [value - 0.5 for value in closes],
            "close": closes,
            "volume": [100] * len(closes),
        }
    )


def test_training_target_uses_the_next_close_and_only_past_features():
    closes = [10, 11, 10, 10.5, 10.2, 10.8, 11, 10.9, 11.2, 11.4, 11.1, 11.8, 11.5]
    features, target, labeled = prepare_training_data(make_candles(closes))

    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == len(closes) - MIN_HISTORY_ROWS
    assert len(target) == len(features)
    assert len(labeled) == len(features)
    assert features.isna().sum().sum() == 0
    assert target.iloc[0] == int(closes[11] > closes[10])
    assert target.iloc[1] == int(closes[12] > closes[11])


def test_normalize_sorts_dates_and_removes_duplicate_timestamps():
    candles = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01", "2024-01-01"],
            "open": [11, 9, 10],
            "high": [12, 10, 11],
            "low": [10, 8, 9],
            "close": [11, 9, 10],
            "volume": [1, 1, 2],
        }
    )

    result = normalize_candles(candles)

    assert len(result) == 2
    assert result["time"].is_monotonic_increasing
    assert result.iloc[0]["close"] == 10


def test_included_brent_dataset_is_ready_for_training():
    dataset_path = Path(__file__).parents[1] / "data" / "brent_prices.csv"
    candles = load_candles(dataset_path)
    features, target, labeled = prepare_training_data(candles)

    assert len(candles) == 1837
    assert len(features) > 1000
    assert len(target) == len(features) == len(labeled)
    assert target.nunique() == 2
    assert features.notna().all().all()
    assert candles["time"].is_monotonic_increasing
    assert not candles["time"].duplicated().any()
