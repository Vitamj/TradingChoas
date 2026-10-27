import numpy as np

from app.features import FEATURE_COLUMNS, MIN_HISTORY_ROWS, build_feature_frame


def sample_candles(count: int = 15) -> list[dict[str, object]]:
    values = [80 + index * 0.15 + ((index % 3) - 1) * 0.2 for index in range(count)]
    return [
        {
            "date": f"2024-01-{index + 1:02d}T00:00:00+00:00",
            "open": value,
            "high": value + 0.5,
            "low": value - 0.5,
            "close": value,
            "volume": 100,
        }
        for index, value in enumerate(values)
    ]


def test_feature_frame_matches_mlflow_model_schema():
    result = build_feature_frame(sample_candles())

    assert list(result.columns) == FEATURE_COLUMNS
    assert result.shape == (1, len(FEATURE_COLUMNS))
    assert np.isfinite(result.to_numpy()).all()


def test_feature_builder_requires_enough_history():
    try:
        build_feature_frame(sample_candles(MIN_HISTORY_ROWS - 1))
    except ValueError as exc:
        assert str(MIN_HISTORY_ROWS) in str(exc)
    else:
        raise AssertionError("Expected a validation error for short history")


def test_feature_builder_rejects_duplicate_dates():
    candles = sample_candles()
    candles[-1]["date"] = candles[-2]["date"]

    try:
        build_feature_frame(candles)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Expected a validation error for duplicate dates")
