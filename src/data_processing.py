# src/data_processing.py

# ==========================
# Imports
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ==========================
# EDA Utilities
# ==========================
def load_csv(file_path):
    """Load CSV file with error handling."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV at {file_path}")
        return None

def numeric_summary_stats(df):
    """Compute summary statistics, skew, and kurtosis for numeric columns."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    summary = df[numeric_cols].describe().T
    summary['skew'] = df[numeric_cols].skew()
    summary['kurtosis'] = df[numeric_cols].kurtosis()
    return summary

def categorical_summary(df):
    """Summary for categorical features."""
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    summary = {}
    for col in categorical_cols:
        summary[col] = {
            "unique": df[col].nunique(),
            "top": df[col].value_counts().idxmax(),
            "freq": df[col].value_counts().max()
        }
    return summary

def plot_numeric_distributions(df):
    """Histograms and boxplots for numeric features."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    try:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols):
            plt.subplot((len(numeric_cols)+3)//4, 4, i+1)
            sns.histplot(df[col], kde=True, bins=30, color='skyblue')
            plt.title(f'Histogram of {col}')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols):
            plt.subplot((len(numeric_cols)+3)//4, 4, i+1)
            sns.boxplot(x=df[col], color='lightgreen')
            plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plotting numeric distributions: {e}")

def plot_categorical_distributions(df):
    """Countplots for categorical features."""
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    try:
        for col in categorical_cols:
            plt.figure(figsize=(10,4))
            sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='pastel')
            plt.title(f'Distribution of {col}')
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error in plotting categorical distributions: {e}")

def correlation_heatmap(df):
    """Heatmap of numeric correlations."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix of Numeric Features")
    plt.show()

def missing_values_report(df):
    """Report missing values and percentages."""
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_percent})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing %", ascending=False)
    return missing_df

def outlier_boxplots(df):
    """Boxplots to detect outliers in numeric features."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=(15,10))
    for i, col in enumerate(numeric_cols):
        plt.subplot((len(numeric_cols)+3)//4,4,i+1)
        sns.boxplot(x=df[col], color='lightcoral')
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# ==========================
# Task-3 Feature Engineering
# ==========================

# ----- Step 1-4: RFM Aggregation -----
class RFMTransformer(BaseEstimator, TransformerMixin):
    """Aggregate RFM features per customer."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        rfm = X_copy.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (X_copy['TransactionStartTime'].max() - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            'Amount': ['sum', 'mean', 'std']  # MonetarySum, MonetaryAvg, MonetaryStd
        })
        rfm.columns = ['Recency', 'Frequency', 'MonetarySum', 'MonetaryAvg', 'MonetaryStd']
        rfm.reset_index(inplace=True)
        return rfm

# ----- Step 5-8: Temporal Features -----
class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Extract temporal features from TransactionStartTime."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        X_copy['Hour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['Day'] = X_copy['TransactionStartTime'].dt.day
        X_copy['Month'] = X_copy['TransactionStartTime'].dt.month
        X_copy['Year'] = X_copy['TransactionStartTime'].dt.year
        return X_copy

# ----- Step 9: K-Means Proxy Variable -----
class KMeansProxy(BaseEstimator, TransformerMixin):
    """Apply K-Means clustering to define proxy default variable."""
    def __init__(self, n_clusters=2, features=None):
        self.n_clusters = n_clusters
        self.features = features
        self.km = None

    def fit(self, X, y=None):
        if self.features is None:
            self.features = ['Recency', 'Frequency', 'MonetarySum']
        self.km = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.km.fit(X[self.features])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['ProxyDefault'] = self.km.predict(X_copy[self.features])
        return X_copy

# ----- Step 10-14: Preprocessing Pipeline with Manual WoE -----
class WoEEncoderManual(BaseEstimator, TransformerMixin):
    """Manual WoE transformation for categorical features."""
    def __init__(self, features, target='ProxyDefault'):
        self.features = features
        self.target = target
        self.woe_dict = {}

    def fit(self, X, y=None):
        df = X.copy()
        for col in self.features:
            df_agg = df.groupby(col)[self.target].agg(['count','sum'])
            df_agg['non_event'] = df_agg['count'] - df_agg['sum']
            df_agg['event_rate'] = df_agg['sum'] / df_agg['sum'].sum()
            df_agg['non_event_rate'] = df_agg['non_event'] / df_agg['non_event'].sum()
            df_agg['woe'] = np.log((df_agg['event_rate'] + 0.0001) / (df_agg['non_event_rate'] + 0.0001))
            self.woe_dict[col] = df_agg['woe'].to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.features:
            df[col] = df[col].map(self.woe_dict[col])
        return df

def preprocess_pipeline(numeric_features, categorical_features, wo_features=None):
    """Pipeline for numeric/categorical preprocessing."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Step 10
        ('scaler', StandardScaler())                    # Step 11
    ])

    if wo_features:
        cat_transformer = WoEEncoderManual(features=wo_features)  # Step 12-13
    else:
        cat_transformer = OneHotEncoder(drop='first', sparse_output=False)
        # the was error here onehotencoder does not have sparse parameter in latest sklearn
        # Step 12

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough'  # keep other columns generated by pipeline
)

    return preprocessor

# ----- Full Task-3 Pipeline -----
def task3_feature_engineering_pipeline(
    df: pd.DataFrame,
    numeric_features=None,
    categorical_features=None,
    woe_features=None
):
    df = df.copy()

    # Ensure datetime
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Reference date
    reference_date = df["TransactionStartTime"].max()

    # -------- RFM FEATURES --------
    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (reference_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            MonetarySum=("Amount", "sum"),
            MonetaryAvg=("Amount", "mean"),
            MonetaryStd=("Amount", "std"),
        )
        .reset_index()
    )

    # Fill NaN std (single transaction customers)
    rfm["MonetaryStd"] = rfm["MonetaryStd"].fillna(0)

    return rfm

# Example usage:
# df = load_csv('path_to_transactions.csv')
# numeric_features = ['Recency', 'Frequency', 'MonetarySum', 'MonetaryAvg', 'MonetaryStd', 'Hour', 'Day', 'Month', 'Year']
# categorical_features = ['SomeCategoricalFeature1', 'SomeCategoricalFeature2']
# wo_features = ['SomeCategoricalFeature1', 'SomeCategoricalFeature2']
# df_processed = task3_feature_engineering_pipeline(df, numeric_features, categorical_features, wo_features)
# ==========================
# End of src/data_processing.py
# ==========================
