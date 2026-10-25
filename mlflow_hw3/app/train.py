import json
import logging
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.data import load_candles, resolve_data_path
from app.eda import save_eda_artifacts
from app.features import FEATURE_COLUMNS, prepare_training_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
EXPERIMENT_NAME = "brent-direction"
MODEL_NAME = os.getenv("MODEL_NAME", "BrentDirectionModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATASET_SOURCE_URL = (
    "https://www.kaggle.com/code/nikitamanaenkov/financial-performance-analysis-brent-and-wti/input"
)
RANDOM_STATE = 42


def log_eda(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    source: Path,
    raw_dataset,
) -> None:
    with mlflow.start_run(run_name="eda"):
        mlflow.set_tag("stage", "exploratory_data_analysis")
        mlflow.set_tag("dataset.source_url", DATASET_SOURCE_URL)
        mlflow.set_tag("dataset.digest", raw_dataset.digest)
        mlflow.log_input(raw_dataset, context="eda")
        mlflow.log_params(
            {
                "dataset_name": "brent-daily-prices",
                "source": str(source.resolve()),
                "rows_after_cleaning": len(candles),
                "feature_count": len(FEATURE_COLUMNS),
                "target_definition": "next_close_greater_than_current_close",
            }
        )
        mlflow.log_metrics(
            {
                "rows_after_cleaning": float(len(candles)),
                "labeled_rows": float(len(target)),
                "target_up_share": float(target.mean()),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_eda_artifacts(candles, features, target, source, output_dir)
            mlflow.log_artifacts(str(output_dir), artifact_path="eda")
        logger.info("eda_artifacts_logged source=%s rows=%s", source, len(candles))


def model_candidates() -> dict[str, object]:
    return {
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_metrics(
    model, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, float], np.ndarray, np.ndarray | None]:
    predictions = np.asarray(model.predict(features)).reshape(-1)
    probabilities = None
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba") and target.nunique() == 2:
        probabilities = np.asarray(model.predict_proba(features))[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(target, probabilities))
    return metrics, predictions, probabilities


def save_run_artifacts(
    model,
    model_name: str,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    output_dir: Path,
    timestamps: pd.Series | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    prediction_data: dict[str, object] = {
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    }
    if timestamps is not None:
        prediction_data["date"] = timestamps.astype(str).to_numpy()
    pd.DataFrame(prediction_data).to_csv(output_dir / "predictions.csv", index=False)

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["down_or_unchanged", "up"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(model_name)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(figure)

    classifier = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
    importance = None
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_).mean(axis=0)
    if importance is not None:
        importance_frame = pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "importance": np.asarray(importance).reshape(-1)}
        ).sort_values("importance", ascending=False)
        importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.barh(importance_frame["feature"], importance_frame["importance"])
        axis.invert_yaxis()
        axis.set_title("Feature importance")
        figure.tight_layout()
        figure.savefig(output_dir / "feature_importance.png", dpi=140)
        plt.close(figure)

    if probabilities is not None:
        pd.DataFrame({"actual": y_test.to_numpy(), "probability_up": probabilities}).to_csv(
            output_dir / "predicted_probabilities.csv", index=False
        )


def select_champion(candidates: list[dict[str, object]]) -> dict[str, object]:
    return max(
        candidates,
        key=lambda item: (float(item["balanced_accuracy"]), float(item["f1"])),
    )


def register_model_version(client: MlflowClient, model_info, run_id: str) -> str:
    registered_version = getattr(model_info, "registered_model_version", None)
    if registered_version is not None:
        return str(registered_version)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    matching = [version for version in versions if version.run_id == run_id]
    if not matching:
        raise RuntimeError(f"Could not find registered version for run {run_id}")
    return str(max(matching, key=lambda version: int(version.version)).version)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    source = resolve_data_path()
    candles = load_candles(source)
    features, target, labeled_candles = prepare_training_data(candles)
    if len(features) < 100:
        raise ValueError(f"At least 100 labeled rows are required, got {len(features)}")

    split_index = int(len(features) * 0.8)
    train_end = split_index - 1
    X_train = features.iloc[:train_end]
    X_test = features.iloc[split_index:]
    y_train = target.iloc[:train_end]
    y_test = target.iloc[split_index:]
    if y_train.nunique() < 2:
        raise ValueError("The training split must contain both target classes")
    if y_test.nunique() < 2:
        raise ValueError("The test split must contain both target classes")

    raw_dataset = mlflow.data.from_pandas(
        candles, source=str(source.resolve()), name="brent-daily-prices"
    )
    feature_data = features.copy()
    feature_data["target"] = target.to_numpy()
    feature_dataset = mlflow.data.from_pandas(
        feature_data, source=str(source.resolve()), name="brent-daily-features"
    )
    log_eda(candles, features, target, source, raw_dataset)

    client = MlflowClient(tracking_uri=TRACKING_URI)
    candidates = []
    comparison_rows = []
    for name, model in model_candidates().items():
        register = name != "baseline_most_frequent"
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("model_family", name)
            mlflow.set_tag("target", "next_daily_close_direction")
            mlflow.set_tag("dataset.source_url", DATASET_SOURCE_URL)
            mlflow.set_tag("dataset.raw_digest", raw_dataset.digest)
            mlflow.set_tag("dataset.feature_digest", feature_dataset.digest)
            mlflow.log_input(raw_dataset, context="source_data")
            mlflow.log_input(feature_dataset, context="training")
            mlflow.log_params(
                {
                    "model_name": name,
                    "dataset_rows": len(candles),
                    "split_strategy": "chronological_80_20_with_one_row_gap",
                    "gap_rows": 1,
                    "random_state": RANDOM_STATE,
                    "feature_count": len(FEATURE_COLUMNS),
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "positive_rate_train": float(y_train.mean()),
                    "target_definition": "next_close_greater_than_current_close",
                    "selection_metric": "balanced_accuracy",
                }
            )
            if name == "logistic_regression":
                mlflow.log_params({"class_weight": "balanced", "max_iter": 1000, "C": 1.0})
            elif name == "random_forest":
                mlflow.log_params(
                    {
                        "n_estimators": 300,
                        "max_depth": 8,
                        "min_samples_leaf": 5,
                        "class_weight": "balanced_subsample",
                    }
                )
            model.fit(X_train, y_train)
            metrics, predictions, probabilities = calculate_metrics(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as directory:
                artifact_dir = Path(directory)
                test_times = (
                    labeled_candles["time"].iloc[split_index] if "time" in labeled_candles else None
                )
                timestamps = (
                    labeled_candles["time"].iloc[split_index:].reset_index(drop=True)
                    if test_times is not None
                    else None
                )
                save_run_artifacts(
                    model,
                    name,
                    y_test,
                    predictions,
                    probabilities,
                    artifact_dir,
                    timestamps,
                )
                mlflow.log_artifacts(str(artifact_dir), artifact_path="diagnostics")
            signature = infer_signature(X_train.head(5), model.predict(X_train.head(5)))
            log_options = {
                "sk_model": model,
                "artifact_path": "model",
                "input_example": X_train.head(3),
                "signature": signature,
            }
            if register:
                log_options["registered_model_name"] = MODEL_NAME
            model_info = mlflow.sklearn.log_model(**log_options)
            version = None
            if register:
                version = register_model_version(client, model_info, run.info.run_id)
                candidates.append(
                    {
                        "name": name,
                        "f1": metrics["f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "version": version,
                        "run_id": run.info.run_id,
                    }
                )
            comparison_rows.append(
                {
                    "model": name,
                    **metrics,
                    "registered_version": version or "not_registered",
                    "run_id": run.info.run_id,
                }
            )
            logger.info("run_finished name=%s metrics=%s", name, metrics)

    if not candidates:
        raise RuntimeError("No candidate models were registered")
    best = select_champion(candidates)
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, best["version"])

    with mlflow.start_run(run_name="model-comparison"):
        mlflow.set_tag("stage", "model_comparison")
        mlflow.set_tag("dataset.source_url", DATASET_SOURCE_URL)
        mlflow.log_input(raw_dataset, context="comparison")
        mlflow.log_input(feature_dataset, context="comparison")
        with tempfile.TemporaryDirectory() as directory:
            comparison_path = Path(directory) / "model_comparison.csv"
            pd.DataFrame(comparison_rows).sort_values("balanced_accuracy", ascending=False).to_csv(
                comparison_path, index=False
            )
            mlflow.log_artifact(str(comparison_path))
        mlflow.log_metrics(
            {
                "champion_f1": float(best["f1"]),
                "champion_balanced_accuracy": float(best["balanced_accuracy"]),
            }
        )
        mlflow.set_tag("champion.version", best["version"])
        mlflow.set_tag("champion.run_id", best["run_id"])

    logger.info(
        "champion_updated model=%s alias=%s version=%s run_id=%s balanced_accuracy=%.4f f1=%.4f",
        MODEL_NAME,
        MODEL_ALIAS,
        best["version"],
        best["run_id"],
        best["balanced_accuracy"],
        best["f1"],
    )
    print(
        f"Champion: {MODEL_NAME} alias={MODEL_ALIAS} version={best['version']} "
        f"run_id={best['run_id']} balanced_accuracy={best['balanced_accuracy']:.4f} "
        f"f1={best['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
