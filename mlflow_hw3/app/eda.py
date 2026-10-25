import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from app.data import data_quality_report


def save_eda_artifacts(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    source: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candles.describe(include="all").transpose().to_csv(output_dir / "summary_statistics.csv")
    features.describe().transpose().to_csv(output_dir / "feature_statistics.csv")

    quality = data_quality_report(source, candles)
    quality["labeled_rows"] = int(len(target))
    quality["target_up_share"] = float(target.mean())
    quality["target_down_or_unchanged_share"] = float(1 - target.mean())
    (output_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    x_axis = candles["time"] if "time" in candles else candles.index
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(x_axis, candles["close"], linewidth=0.8)
    axis.set_title("Brent daily close price")
    axis.set_xlabel("Date")
    axis.set_ylabel("Close price")
    figure.tight_layout()
    figure.savefig(output_dir / "close_price.png", dpi=140)
    plt.close(figure)

    counts = target.value_counts().reindex([0, 1], fill_value=0)
    pd.DataFrame(
        {
            "direction": ["down_or_unchanged", "up"],
            "rows": [int(counts.loc[0]), int(counts.loc[1])],
        }
    ).to_csv(output_dir / "target_distribution.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Down or unchanged", "Up"], counts.values)
    axis.set_title("Next daily close direction")
    axis.set_ylabel("Rows")
    figure.tight_layout()
    figure.savefig(output_dir / "target_distribution.png", dpi=140)
    plt.close(figure)

    returns = candles["close"].pct_change(fill_method=None).dropna()
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(returns, bins=50)
    axis.set_title("Distribution of daily close returns")
    axis.set_xlabel("Daily return")
    axis.set_ylabel("Rows")
    figure.tight_layout()
    figure.savefig(output_dir / "return_distribution.png", dpi=140)
    plt.close(figure)

    rolling_volatility = returns.rolling(10).std(ddof=0)
    rolling_x_axis = candles["time"].iloc[1:] if "time" in candles else candles.index[1:]
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(rolling_x_axis, rolling_volatility, linewidth=0.8)
    axis.set_title("10-day rolling volatility")
    axis.set_xlabel("Date")
    axis.set_ylabel("Standard deviation of daily returns")
    figure.tight_layout()
    figure.savefig(output_dir / "rolling_volatility.png", dpi=140)
    plt.close(figure)

    correlation = features.assign(target=target.to_numpy()).corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=70, ha="right")
    axis.set_yticks(range(len(correlation.index)), correlation.index)
    figure.colorbar(image, ax=axis, shrink=0.8)
    axis.set_title("Feature correlation matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "feature_correlation.png", dpi=140)
    plt.close(figure)
