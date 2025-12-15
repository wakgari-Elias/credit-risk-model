import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/processed/data_with_target.csv"
EXPERIMENT_NAME = "credit-risk-task-5"
MODEL_NAME = "credit-risk-model"   # 🔴 MUST MATCH API
RANDOM_STATE = 42


# -----------------------------
# Utility functions
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop IDs
    drop_cols = ["BatchId", "CustomerId", "TransactionId", "TransactionStartTime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categoricals
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# -----------------------------
# Training functions
# -----------------------------
def train_logistic_regression(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="Logistic_Regression"):
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        return metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    with mlflow.start_run(run_name="Random_Forest_GridSearch"):
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        return metrics


# -----------------------------
# Promote best model
# -----------------------------
def promote_latest_to_production():
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME)

    latest_version = max(int(v.version) for v in versions)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"✅ Model v{latest_version} promoted to Production")


# -----------------------------
# Main
# -----------------------------
def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(DATA_PATH)
    df = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("Training Logistic Regression...")
    lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
    print("LR metrics:", lr_metrics)

    print("Training Random Forest...")
    rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    print("RF metrics:", rf_metrics)

    promote_latest_to_production()


if __name__ == "__main__":
    main()
