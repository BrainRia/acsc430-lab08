from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use("Agg")


@dataclass
class ModelResult:
    feature_name: str
    linear_model: LinearRegression
    knn_model: KNeighborsClassifier
    linear_score: float
    knn_score: float
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    knn_accuracy: float


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset: {csv_path}. Place fraud.csv next to Lab08.py."
        )

    data = pd.read_csv(csv_path)
    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    return data.dropna().copy()


def find_good_v_relationships(
    data: pd.DataFrame, min_abs_corr: float = 0.70, limit: int = 5
) -> list[tuple[str, str, float]]:
    v_columns = [column for column in data.columns if column.startswith("V")]
    pairs: list[tuple[str, str, float]] = []

    for left, right in combinations(v_columns, 2):
        correlation = data[left].corr(data[right])
        if pd.notna(correlation):
            pairs.append((left, right, correlation))

    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    strong_pairs = [pair for pair in pairs if abs(pair[2]) >= min_abs_corr]
    return strong_pairs[:limit] if strong_pairs else pairs[:limit]


def train_models_for_feature(
    data: pd.DataFrame,
    feature_name: str,
    *,
    random_state: int = 42,
    test_size: float = 0.25,
    knn_neighbors: int = 5,
) -> ModelResult:
    x = data[[feature_name]]
    y = data["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    linear_score = linear_model.score(x_test, y_test)

    knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn_model.fit(x_train, y_train)
    knn_predictions = knn_model.predict(x_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    knn_score = knn_model.score(x_test, y_test)

    return ModelResult(
        feature_name=feature_name,
        linear_model=linear_model,
        knn_model=knn_model,
        linear_score=linear_score,
        knn_score=knn_score,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        knn_accuracy=knn_accuracy,
    )


def choose_feature(data: pd.DataFrame, baseline_feature: str = "V1") -> tuple[ModelResult, ModelResult]:
    if baseline_feature not in data.columns:
        raise ValueError(f"Baseline feature '{baseline_feature}' was not found in the dataset.")

    candidate_features = [
        column
        for column in data.columns
        if column != "Class" and column != baseline_feature
    ]

    baseline_result = train_models_for_feature(data, baseline_feature)
    candidate_results = [train_models_for_feature(data, feature) for feature in candidate_features]

    best_result = max(
        candidate_results,
        key=lambda result: (result.knn_score, result.linear_score),
    )

    if (
        best_result.knn_score <= baseline_result.knn_score
        and best_result.linear_score <= baseline_result.linear_score
    ):
        best_result = max(
            candidate_results,
            key=lambda result: (
                result.knn_score - baseline_result.knn_score,
                result.linear_score - baseline_result.linear_score,
            ),
        )

    return baseline_result, best_result


def describe_linear_regression(model: LinearRegression, feature_name: str) -> str:
    coefficient = float(model.coef_[0])
    intercept = float(model.intercept_)
    return f"Class = {coefficient:.6f} * {feature_name} + {intercept:.6f}"


def describe_knn_model(model: KNeighborsClassifier, feature_name: str) -> str:
    return (
        f"predict_class({feature_name}) = majority_class_of_{model.n_neighbors}_nearest_training_points"
    )


def print_relationships(relationships: Iterable[tuple[str, str, float]]) -> None:
    print("Good relationships between Vi and Vj:")
    for left, right, correlation in relationships:
        print(f"  {left} - {right}: correlation = {correlation:.4f}")


def create_comparison_plot(
    baseline_result: ModelResult,
    chosen_result: ModelResult,
    output_path: Path,
) -> None:
    labels = [
        f"Linear ({baseline_result.feature_name})",
        f"KNN ({baseline_result.feature_name})",
        f"Linear ({chosen_result.feature_name})",
        f"KNN ({chosen_result.feature_name})",
    ]
    scores = [
        baseline_result.linear_score,
        baseline_result.knn_score,
        chosen_result.linear_score,
        chosen_result.knn_score,
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, scores, color=colors)
    plt.ylabel("Score")
    plt.title("Performance Comparison for V1 and the Chosen Feature")
    plt.ylim(min(min(scores), 0) - 0.05, max(scores) + 0.05)
    plt.xticks(rotation=10)

    for bar, score in zip(bars, scores, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
