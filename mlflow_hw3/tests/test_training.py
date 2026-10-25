import numpy as np
import pandas as pd

from app.train import calculate_metrics, model_candidates, select_champion


def test_experiment_contains_baseline_and_two_model_families():
    candidates = model_candidates()

    assert set(candidates) == {
        "baseline_most_frequent",
        "logistic_regression",
        "random_forest",
    }


def test_calculate_metrics_returns_primary_and_supporting_metrics():
    model = model_candidates()["baseline_most_frequent"]
    features = pd.DataFrame({"return_1": [0.01, -0.02, 0.03, -0.01]})
    target = pd.Series([1, 0, 1, 0])
    model.fit(features, target)

    metrics, predictions, probabilities = calculate_metrics(model, features, target)

    assert {"balanced_accuracy", "f1", "accuracy", "precision", "recall", "roc_auc"} <= set(metrics)
    assert len(predictions) == len(target)
    assert probabilities is not None
    assert np.isclose(metrics["balanced_accuracy"], 0.5)


def test_champion_selection_uses_balanced_accuracy_before_f1():
    candidates = [
        {"name": "baseline", "balanced_accuracy": 0.5, "f1": 0.66},
        {"name": "random_forest", "balanced_accuracy": 0.52, "f1": 0.53},
        {"name": "logistic_regression", "balanced_accuracy": 0.51, "f1": 0.55},
    ]

    selected = select_champion(candidates)

    assert selected["name"] == "random_forest"
