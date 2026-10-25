import pandas as pd

from app.data import normalize_candles

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


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_candles(candles)
    close = frame["close"].astype(float)
    returns = close.pct_change(fill_method=None)
    features = pd.DataFrame(index=frame.index)
    for period in (1, 2, 3, 5, 10):
        features[f"return_{period}"] = close.pct_change(periods=period, fill_method=None)
    features["volatility_5"] = returns.rolling(5).std(ddof=0)
    features["volatility_10"] = returns.rolling(10).std(ddof=0)
    features["sma_gap_5"] = close / close.rolling(5).mean() - 1
    features["sma_gap_10"] = close / close.rolling(10).mean() - 1
    return features[FEATURE_COLUMNS].reset_index(drop=True)


def prepare_training_data(
    candles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    frame = normalize_candles(candles)
    features = build_features(frame)
    next_close = frame["close"].shift(-1)
    target = (next_close > frame["close"]).astype("int8")
    valid_rows = next_close.notna() & features.notna().all(axis=1)
    return (
        features.loc[valid_rows].reset_index(drop=True),
        target.loc[valid_rows].reset_index(drop=True),
        frame.loc[valid_rows].reset_index(drop=True),
    )
