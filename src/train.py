"""Train, cross-validate, compare, evaluate and log classifiers with MLflow."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.data import DATA_PATH, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
RANDOM_STATE = 42


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([("numeric", numeric_pipe, feature_names)], remainder="drop")


def make_models(feature_names: list[str]) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor(feature_names)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor(feature_names)),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
    }


def _safe_auc(y_true, probability) -> float:
    try:
        return float(roc_auc_score(y_true, probability))
    except ValueError:
        return float("nan")


def train(data_path: Path = DATA_PATH, tune: bool = True) -> dict:
    frame = load_dataset(data_path)
    frame = frame.dropna(subset=["target"]).copy()
    frame["target"] = frame["target"].astype(int)
    X = frame.drop(columns="target")
    y = frame["target"]
    if y.nunique() != 2:
        raise ValueError("Target must contain exactly two classes.")
    if y.value_counts().min() < 5:
        warnings.warn("A target class has fewer than 5 rows; 5-fold CV may be unreliable.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("uci-heart-disease-classification")

    results = {}
    fitted_models = {}
    for name, pipeline in make_models(list(X.columns)).items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model", name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("feature_count", X.shape[1])
            cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            cv_summary = {
                metric: float(np.mean(cv_scores[f"test_{metric}"]))
                for metric in scoring
            }
            for metric, value in cv_summary.items():
                mlflow.log_metric(f"cv_{metric}_mean", value)

            # Small, explicit tuning grid for Random Forest; CV stays inside training data.
            if tune and name == "random_forest":
                search = GridSearchCV(
                    pipeline,
                    {"model__max_depth": [None, 4, 8], "model__min_samples_leaf": [1, 2, 4]},
                    scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
                )
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
                mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
                mlflow.log_metric("tuned_cv_roc_auc", float(search.best_score_))
            else:
                fitted = pipeline.fit(X_train, y_train)

            pred = fitted.predict(X_test)
            proba = fitted.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": _safe_auc(y_test, proba),
                "cv": cv_summary,
            }
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                mlflow.log_metric(f"test_{key}", metrics[key])
            mlflow.log_dict(metrics, f"{name}_metrics.json")
            mlflow.sklearn.log_model(fitted, artifact_path="model")
            results[name] = {"metrics": metrics, "run_id": run.info.run_id}
            fitted_models[name] = fitted

    best_name = max(results, key=lambda n: results[n]["metrics"]["cv"]["roc_auc"])
    best_model = fitted_models[best_name]
    best_pred = best_model.predict(X_test)
    best_proba = best_model.predict_proba(X_test)[:, 1]
    final_metrics = {
        "selected_model": best_name,
        **{k: v for k, v in results[best_name]["metrics"].items() if k != "cv"},
        "cv": results[best_name]["metrics"]["cv"],
        "classification_report": classification_report(y_test, best_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
        "test_rows": int(len(y_test)),
        "feature_names": list(X.columns),
        "note": "Educational dataset benchmark; not a medical diagnostic system.",
    }
    joblib.dump(best_model, MODEL_DIR / "heart_disease_pipeline.joblib")
    metadata = {
        "selected_model": best_name,
        "feature_names": list(X.columns),
        "target": "target",
        "positive_class_meaning": "UCI target > 0 (presence of heart disease)",
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (REPORT_DIR / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
    (REPORT_DIR / "cv_results.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, best_proba, ax=ax)
    ax.set_title(f"ROC curve — {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title(f"Confusion matrix — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(json.dumps(final_metrics, indent=2))
    print(f"Saved model: {MODEL_DIR / 'heart_disease_pipeline.joblib'}")
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--no-tune", action="store_true", help="Skip Random Forest grid search")
    args = parser.parse_args()
    train(args.data, tune=not args.no_tune)
