from __future__ import annotations

from pathlib import Path

import pandas as pd

from lab08_utils import (
    choose_feature,
    create_comparison_plot,
    describe_knn_model,
    describe_linear_regression,
    find_good_v_relationships,
    load_data,
    print_relationships,
)

def print_dataset_overview(data: pd.DataFrame) -> None:
    print("Dataset overview")
    print(f"  Rows: {data.shape[0]}")
    print(f"  Columns: {data.shape[1]}")
    print(f"  Target column: Class")
    print()
    print("First 5 rows:")
    print(data.head())
    print()
def print_model_results(title: str, result) -> None:
    print(title)
    print(f"  Input feature (x): {result.feature_name}")
    print(f"  Output feature (y): Class")
    print(f"  Training samples: {len(result.x_train)}")
    print(f"  Test samples: {len(result.x_test)}")
    print()
    print("  Linear Regression")
    print(f"    Score: {result.linear_score:.6f}")
    print(f"    Function: {describe_linear_regression(result.linear_model, result.feature_name)}")
    print()
    print("  KNN Classification")
    print(f"    Score: {result.knn_score:.6f}")
    print(f"    Accuracy: {result.knn_accuracy:.6f}")
    print(f"    Function: {describe_knn_model(result.knn_model, result.feature_name)}")
    print()
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "fraud.csv"
    plot_path = base_dir / "model_comparison.png"
    data = load_data(csv_path)
    print_dataset_overview(data)
    relationships = find_good_v_relationships(data)
    print_relationships(relationships)
    print()
    baseline_result, chosen_result = choose_feature(data, baseline_feature="V1")

    print_model_results("Results for the sample feature V1", baseline_result)
    print_model_results("Results for the chosen feature", chosen_result)

    if chosen_result.feature_name == "V1":
        print("No better single-feature predictor than V1 was found in this run.")
    else:
        print(
            f"Chosen feature: {chosen_result.feature_name} "
            f"(better than V1 mainly on KNN score)."
        )
    print()

    create_comparison_plot(baseline_result, chosen_result, plot_path)
    print(f"Saved comparison plot to: {plot_path}")


if __name__ == "__main__":
    main()
