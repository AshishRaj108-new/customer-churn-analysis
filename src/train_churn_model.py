#!/usr/bin/env python3
"""Compare three churn models and export holdout predictions with SHAP explanations.

All models use the same stratified train/test split and the same preprocessed
features. The best model is selected by held-out ROC-AUC (F1 breaks ties).
Only held-out rows are exported to the static dashboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "data" / "ml_predictions.json"
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three churn classifiers and export holdout results plus SHAP explanations."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the Telco churn CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path for the dashboard.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for holdout evaluation.")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED, help="Seed used for the reproducible split.")
    parser.add_argument("--shap-background-size", type=int, default=200, help="Training rows used as SHAP background data.")
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN, *NUMERIC_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    frame = frame.loc[frame[TARGET_COLUMN].isin(["Yes", "No"])].copy()
    if frame.empty:
        raise ValueError("No rows with a Churn value of Yes or No were found.")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    identifiers = frame[ID_COLUMN].astype(str)
    labels = frame[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    features = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    return features, labels, identifiers


def make_dense_encoder() -> OneHotEncoder:
    """Keep a dense matrix so all three classifiers can use identical input."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn before 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric = [column for column in NUMERIC_COLUMNS if column in feature_columns]
    categorical = [column for column in feature_columns if column not in numeric]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_dense_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        verbose_feature_names_out=False,
    )


def build_models(random_state: int) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=random_state,
        ),
    }


def evaluate_model(model: Any, x_test: np.ndarray, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusionMatrix": {
            "trueNegative": int(tn),
            "falsePositive": int(fp),
            "falseNegative": int(fn),
            "truePositive": int(tp),
        },
    }
    return probabilities, predictions, metrics


def positive_class_shap_values(explanation: Any) -> np.ndarray:
    """Return SHAP values for the positive (Churn=Yes) class across SHAP versions."""
    values = explanation.values if hasattr(explanation, "values") else explanation
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    return values


def grouped_feature_indices(preprocessor: ColumnTransformer, feature_columns: list[str]) -> list[tuple[str, list[int]]]:
    numeric = [column for column in NUMERIC_COLUMNS if column in feature_columns]
    categorical = [column for column in feature_columns if column not in numeric]
    groups: list[tuple[str, list[int]]] = []

    offset = 0
    for column in numeric:
        groups.append((column, [offset]))
        offset += 1

    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["onehot"]
    for column, categories in zip(categorical, encoder.categories_):
        indices = list(range(offset, offset + len(categories)))
        groups.append((column, indices))
        offset += len(categories)
    return groups


def human_value(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):g}"
    return str(value)


def summarize_explanations(
    shap_values: np.ndarray,
    test_features: pd.DataFrame,
    groups: list[tuple[str, list[int]]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, str]]]]:
    grouped = np.column_stack([shap_values[:, indices].sum(axis=1) for _, indices in groups])
    global_importance = np.mean(np.abs(grouped), axis=0)
    global_order = np.argsort(global_importance)[::-1][:6]
    global_drivers = [
        {
            "feature": groups[index][0],
            "meanAbsContribution": round(float(global_importance[index]), 5),
        }
        for index in global_order
    ]

    row_reasons: list[list[dict[str, str]]] = []
    for row_number, contributions in enumerate(grouped):
        order = np.argsort(np.abs(contributions))[::-1][:3]
        reasons = []
        for index in order:
            contribution = float(contributions[index])
            if abs(contribution) < 1e-8:
                continue
            feature = groups[index][0]
            reasons.append(
                {
                    "feature": feature,
                    "value": human_value(test_features.iloc[row_number][feature]),
                    "direction": "raises" if contribution > 0 else "lowers",
                }
            )
        row_reasons.append(reasons)
    return global_drivers, row_reasons


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be greater than 0 and less than 1.")

    features, labels, identifiers = load_dataset(args.csv)
    feature_columns = list(features.columns)
    x_train, x_test, y_train, y_test, ids_test = train_test_split(
        features,
        labels,
        identifiers,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    preprocessor = build_preprocessor(feature_columns)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)
    background_size = min(max(1, args.shap_background_size), x_train_transformed.shape[0])
    background = x_train_transformed[:background_size]

    fitted_models: dict[str, Any] = {}
    model_results: dict[str, dict[str, Any]] = {}
    for model_name, model in build_models(args.random_state).items():
        model.fit(x_train_transformed, y_train)
        probabilities, predictions, metrics = evaluate_model(model, x_test_transformed, y_test)
        fitted_models[model_name] = model
        model_results[model_name] = {
            "probabilities": probabilities,
            "predictions": predictions,
            "metrics": metrics,
        }

    best_name = max(
        model_results,
        key=lambda name: (
            model_results[name]["metrics"]["rocAuc"],
            model_results[name]["metrics"]["f1"],
            model_results[name]["metrics"]["accuracy"],
        ),
    )
    best_model = fitted_models[best_name]
    best_result = model_results[best_name]

    if best_name == "Logistic Regression":
        explainer = shap.LinearExplainer(best_model, background)
    else:
        explainer = shap.TreeExplainer(best_model, data=background)
    shap_values = positive_class_shap_values(explainer(x_test_transformed))
    groups = grouped_feature_indices(preprocessor, feature_columns)
    global_drivers, row_reasons = summarize_explanations(shap_values, x_test, groups)

    comparison = [
        {
            "model": name,
            "selected": name == best_name,
            "metrics": result["metrics"],
        }
        for name, result in sorted(
            model_results.items(),
            key=lambda item: (
                item[1]["metrics"]["rocAuc"],
                item[1]["metrics"]["f1"],
            ),
            reverse=True,
        )
    ]
    output = {
        "schemaVersion": 2,
        "model": best_name,
        "target": "Churn",
        "positiveClass": "Yes",
        "threshold": 0.5,
        "selectionRule": "Highest held-out ROC-AUC; F1 score breaks ties.",
        "modelComparison": comparison,
        "evaluation": {
            "method": f"{args.test_size:.0%} stratified holdout test set",
            "datasetRows": int(len(features)),
            "trainingRows": int(len(x_train)),
            "testRows": int(len(x_test)),
            "metrics": best_result["metrics"],
            "globalDrivers": global_drivers,
        },
        "explanationNote": "SHAP explains how recorded features influenced this model score; these associations do not prove cause.",
        "predictions": [
            {
                "customerID": str(customer_id),
                "churnProbability": round(float(probability), 5),
                "predictedChurn": "Yes" if prediction else "No",
                "observedChurn": "Yes" if actual else "No",
                "topReasons": reasons,
            }
            for customer_id, probability, prediction, actual, reasons in zip(
                ids_test,
                best_result["probabilities"],
                best_result["predictions"],
                y_test.to_numpy(),
                row_reasons,
            )
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Model comparison (same held-out test set):")
    for result in comparison:
        metrics = result["metrics"]
        marker = "  <-- selected" if result["selected"] else ""
        print(
            f"  {result['model']}: ROC-AUC={metrics['rocAuc']:.4f}, "
            f"F1={metrics['f1']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, accuracy={metrics['accuracy']:.4f}{marker}"
        )
    print(f"Training rows: {output['evaluation']['trainingRows']}")
    print(f"Held-out test rows: {output['evaluation']['testRows']}")
    print(f"SHAP reasons generated for {len(row_reasons)} held-out customers.")
    print(f"Dashboard data written to: {args.output}")


if __name__ == "__main__":
    main()
