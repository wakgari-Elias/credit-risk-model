# src/data_processing.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
