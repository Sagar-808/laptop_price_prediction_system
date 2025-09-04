import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scratch pipeline (no sklearn)
try:
    from .scratch_pipeline import Preprocessor, ScratchPipeline  # type: ignore
except Exception:
    import sys as _sys
    import os as _os
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    from scratch_pipeline import Preprocessor, ScratchPipeline  # type: ignore

# Support running as a module or as a script
try:
    from .laptop_preprocess import parse_laptop_dataframe  # type: ignore
except Exception:
    # When executed as `python model/train.py`, relative imports fail.
    # Add this directory to sys.path and import directly.
    import sys as _sys
    import os as _os
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    from laptop_preprocess import parse_laptop_dataframe  # type: ignore


def train_laptop_price_model(csv_path: str, output_path: str) -> dict:
    print("Loading laptop dataset...", csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Build engineered features and split
    X, y, cat_cols, num_cols = parse_laptop_dataframe(df, drop_target=False)

    # Simple train/test split (deterministic shuffle)
    rng = np.random.default_rng(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    # Pipeline: preprocess -> linear regression
    preprocessor = Preprocessor(cat_cols, num_cols)

    # Use scratch implementation instead of importing algorithm
    try:
        from .linear_regression import MyLinearRegression  # type: ignore
    except Exception:
        import sys as _sys
        import os as _os
        _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
        from linear_regression import MyLinearRegression  # type: ignore

    pipeline = ScratchPipeline(preprocessor, MyLinearRegression(fit_intercept=True))

    print("Training Linear Regression pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    # Metrics (scratch)
    def _rmse(y_true, y_pred):
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        return np.sqrt(np.mean((y_t - y_p) ** 2))

    def _mae(y_true, y_pred):
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        return np.mean(np.abs(y_t - y_p))

    def _r2(y_true, y_pred):
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    rmse = float(_rmse(y_test, y_pred_test))
    mae = float(_mae(y_test, y_pred_test))
    r2_test = float(_r2(y_test, y_pred_test))
    r2_train = float(_r2(y_train, y_pred_train))

    # Classification reports and confusion matrix do not apply to regression.
    # For your college report, we include train/test R^2, RMSE, MAE, and a residual summary.
    residuals = (y_test - y_pred_test).tolist()

    # Create evaluation plots (Predicted vs Actual, Residuals histogram)
    out_dir = os.path.dirname(output_path)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Predicted vs Actual scatter
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_test, alpha=0.5, edgecolor='k')
    lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('Actual Price (INR)')
    plt.ylabel('Predicted Price (INR)')
    plt.title('Predicted vs Actual')
    pva_path = os.path.join(plots_dir, 'predicted_vs_actual.png')
    plt.tight_layout()
    plt.savefig(pva_path, dpi=150)
    plt.close()

    # Residuals histogram
    plt.figure(figsize=(6,4))
    plt.hist(y_test - y_pred_test, bins=30, alpha=0.75, color='#4e79a7', edgecolor='k')
    plt.xlabel('Residual (Actual - Predicted) [INR]')
    plt.ylabel('Count')
    plt.title('Residuals Histogram')
    resid_path = os.path.join(plots_dir, 'residuals_hist.png')
    plt.tight_layout()
    plt.savefig(resid_path, dpi=150)
    plt.close()

    metrics = {
        "problem_type": "regression",
        "classification_metrics_applicable": False,
        "confusion_matrix_applicable": False,
        # Error metrics
        "rmse": rmse,
        "mae": mae,
        # R^2 metrics (clear names, with backward-compatible aliases)
        "r2_score_test": r2_test,
        "r2_score_train": r2_train,
        "r2_test": r2_test,     # alias (backwards compatibility)
        "r2_train": r2_train,   # alias (backwards compatibility)
        # Plots
        "plots": {
            "predicted_vs_actual": pva_path,
            "residuals_hist": resid_path,
        },
        # Diagnostics
        "residuals_sample": residuals[:50],
    }
    print("Metrics:", metrics)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
    }
    joblib.dump(payload, output_path)
    print("Saved model to", output_path)

    # Save report
    report_path = output_path.replace(".pkl", "_report.json")
    with open(report_path, "w") as f:
        json.dump(payload | {"pipeline": None}, f, indent=2)
    print("Saved report to", report_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Laptop Price Prediction Model")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to laptop_data.csv (defaults to project root laptop_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output model path (defaults to backend/model/laptop_price_model.pkl)",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    csv_path = args.data or os.path.join(project_root, "laptop_data.csv")
    output_path = args.output or os.path.join(current_dir, "laptop_price_model.pkl")

    train_laptop_price_model(csv_path, output_path)


if __name__ == "__main__":
    main()