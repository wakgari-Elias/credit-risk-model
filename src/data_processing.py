# ==========================
# src/data_processing.py
# Task-2 (EDA Utilities) + Task-3 (Feature Engineering)
# ==========================

# --------------------------
# Imports
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# Task-2: EDA & Data Understanding Utilities
# =====================================================

def load_csv(file_path: str) -> pd.DataFrame | None:
    """Load CSV file with basic error handling."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.ParserError:
        print(f"Parsing error for file: {file_path}")
    return None


def numeric_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics + skewness and kurtosis for numeric columns."""
    num_cols = df.select_dtypes(include="number")
    summary = num_cols.describe().T
    summary["skew"] = num_cols.skew()
    summary["kurtosis"] = num_cols.kurtosis()
    return summary


def categorical_summary(df: pd.DataFrame) -> dict:
    """Return summary for categorical columns."""
    summary = {}
    for col in df.select_dtypes(include="object").columns:
        summary[col] = {
            "unique": df[col].nunique(),
            "top": df[col].value_counts().idxmax(),
            "freq": df[col].value_counts().max(),
        }
    return summary


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing values count and percentage."""
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    report = pd.DataFrame({"Missing Count": missing, "Missing %": percent})
    return report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Plot histograms for numeric features."""
    for col in df.select_dtypes(include="number").columns:
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """Plot count plots for categorical features."""
    for col in df.select_dtypes(include="object").columns:
        sns.countplot(y=col, data=df)
        plt.title(f"Distribution of {col}")
        plt.show()


def correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numeric features."""
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

# =====================================================
# Task-3: Feature Engineering Pipeline
# =====================================================

# --------------------------
# Temporal Feature Extraction
# --------------------------
class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Extract hour, day, month, year from TransactionStartTime."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])
        X["Hour"] = X["TransactionStartTime"].dt.hour
        X["Day"] = X["TransactionStartTime"].dt.day
        X["Month"] = X["TransactionStartTime"].dt.month
        X["Year"] = X["TransactionStartTime"].dt.year
        return X

# --------------------------
# RFM Aggregation
# --------------------------
class RFMTransformer(BaseEstimator, TransformerMixin):
    """Create Recency, Frequency, Monetary features per customer."""

    def fit(self, X, y=None):
        self.reference_date_ = X["TransactionStartTime"].max()
        return self

    def transform(self, X):
        rfm = (
            X.groupby("CustomerId")
            .agg(
                Recency=("TransactionStartTime", lambda x: (self.reference_date_ - x.max()).days),
                Frequency=("TransactionId", "count"),
                MonetarySum=("Amount", "sum"),
                MonetaryAvg=("Amount", "mean"),
                MonetaryStd=("Amount", "std"),
            )
            .reset_index()
        )
        rfm["MonetaryStd"] = rfm["MonetaryStd"].fillna(0)
        return rfm

# --------------------------
# Proxy Default Variable
# --------------------------
class KMeansProxy(BaseEstimator, TransformerMixin):
    """Generate proxy default variable using KMeans."""

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.km_ = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.km_.fit(X[["Recency", "Frequency", "MonetarySum"]])
        return self

    def transform(self, X):
        X = X.copy()
        X["ProxyDefault"] = self.km_.predict(X[["Recency", "Frequency", "MonetarySum"]])
        return X

# --------------------------
# Weight of Evidence Encoder
# --------------------------
class WoEEncoder(BaseEstimator, TransformerMixin):
    """Apply Weight of Evidence encoding and compute Information Value."""

    def __init__(self, features, target="ProxyDefault"):
        self.features = features
        self.target = target
        self.woe_maps_ = {}
        self.iv_ = {}

    def fit(self, X, y=None):
        eps = 1e-4
        for col in self.features:
            grouped = X.groupby(col)[self.target].agg(["count", "sum"])
            grouped["non_event"] = grouped["count"] - grouped["sum"]

            grouped["event_rate"] = grouped["sum"] / grouped["sum"].sum()
            grouped["non_event_rate"] = grouped["non_event"] / grouped["non_event"].sum()

            grouped["woe"] = np.log((grouped["event_rate"] + eps) / (grouped["non_event_rate"] + eps))
            grouped["iv"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped["woe"]

            self.woe_maps_[col] = grouped["woe"].to_dict()
            self.iv_[col] = grouped["iv"].sum()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.features:
            X[col] = X[col].map(self.woe_maps_[col])
        return X

# --------------------------
# Full Task-3 Pipeline
# --------------------------

def build_task3_pipeline() -> Pipeline:
    numeric_features = [
        "Recency",
        "Frequency",
        "MonetarySum",
        "MonetaryAvg",
        "MonetaryStd",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
    ])

    pipeline = Pipeline([
        ("temporal", TemporalFeatures()),
        ("rfm", RFMTransformer()),
        ("proxy", KMeansProxy()),
        ("preprocess", preprocessor),
    ])

    return pipeline


# --------------------------
# Example Usage
# --------------------------
# df = load_csv("transactions.csv")
# pipeline = build_task3_pipeline(categorical_features=["Category"])
# X_model_ready = pipeline.fit_transform(df)

# ==========================
# End of File
# ==========================


# Example usage:
# df = load_csv('path_to_transactions.csv')
# numeric_features = ['Recency', 'Frequency', 'MonetarySum', 'MonetaryAvg', 'MonetaryStd', 'Hour', 'Day', 'Month', 'Year']
# categorical_features = ['SomeCategoricalFeature1', 'SomeCategoricalFeature2']
# wo_features = ['SomeCategoricalFeature1', 'SomeCategoricalFeature2']
# df_processed = task3_feature_engineering_pipeline(df, numeric_features, categorical_features, wo_features)
# ==========================
# End of src/data_processing.py
# ==========================
