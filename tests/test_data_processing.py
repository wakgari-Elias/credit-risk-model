import sys
import os
import pandas as pd
import numpy as np
import unittest

# Make src importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.data_processing import (
    load_csv,
    numeric_summary_stats,
    categorical_summary,
    missing_values_report,
    build_task3_pipeline,
    calculate_rfm,
    create_target_variable
)


class TestDataProcessing(unittest.TestCase):

    # -----------------------------
    # Task-2 Sample Data
    # -----------------------------
    def setUp(self):
        self.df_task2 = pd.DataFrame({
            "Amount": [10, 20, 30],
            "Value": [100, 200, 300],
            "Category": ["A", "B", "A"],
        })

        # -----------------------------
        # Task-3 Sample Transaction Data
        # -----------------------------
        self.df_task3 = pd.DataFrame({
            "CustomerId": [1, 1, 2, 2, 3, 3],
            "TransactionId": [101, 102, 201, 202, 301, 302],
            "TransactionStartTime": [
                "2025-12-10 08:00:00",
                "2025-12-12 12:00:00",
                "2025-12-11 09:30:00",
                "2025-12-13 14:45:00",
                "2025-12-12 10:15:00",
                "2025-12-14 16:30:00",
            ],
            "Amount": [100, 150, 200, 250, 300, 350],
            "Category": ["A", "B", "A", "B", "A", "B"],
        })

    # =====================================================
    # Task-2 Tests (EDA)
    # =====================================================
    def test_numeric_summary_stats(self):
        result = numeric_summary_stats(self.df_task2)
        self.assertIn("skew", result.columns)
        self.assertIn("kurtosis", result.columns)
        self.assertEqual(result.shape[0], 2)

    def test_categorical_summary(self):
        result = categorical_summary(self.df_task2)
        self.assertIn("Category", result)
        self.assertEqual(result["Category"]["top"], "A")

    def test_missing_values_report(self):
        df = self.df_task2.copy()
        df.loc[0, "Amount"] = np.nan
        report = missing_values_report(df)
        self.assertIn("Amount", report.index)

    # =====================================================
    # Task-3 Tests (Feature Engineering)
    # =====================================================
    def test_task3_pipeline_runs(self):
        pipeline = build_task3_pipeline()
        X_transformed = pipeline.fit_transform(self.df_task3)
        self.assertIsNotNone(X_transformed)

    def test_rfm_feature_dimensions(self):
        pipeline = build_task3_pipeline()
        X_transformed = pipeline.fit_transform(self.df_task3)
        n_customers = self.df_task3["CustomerId"].nunique()
        self.assertEqual(X_transformed.shape[0], n_customers)

    def test_pipeline_output_is_numeric(self):
        pipeline = build_task3_pipeline()
        X_transformed = pipeline.fit_transform(self.df_task3)
        self.assertTrue(np.issubdtype(X_transformed.dtype, np.number))

    # =====================================================
    # Task-5 Unit Tests (REQUIRED)
    # =====================================================
    def test_rfm_columns(self):
        df = pd.DataFrame({
            "CustomerId": [1, 2],
            "TransactionId": [101, 102],
            "Amount": [100, 200],
            "TransactionStartTime": ["2025-01-01", "2025-01-02"],
        })
        rfm = calculate_rfm(df)
        expected_cols = ["CustomerId", "Recency", "Frequency", "Monetary"]
        self.assertTrue(all(col in rfm.columns for col in expected_cols))

    def test_high_risk_column(self):
        # create RFM features
        df = pd.DataFrame({
            "CustomerId": [1, 2],
            "TransactionId": [101, 102],
            "Amount": [100, 200],
            "TransactionStartTime": ["2025-01-01", "2025-01-02"],
        })
        rfm = calculate_rfm(df)

        # 🔧 FIX: mock the clustering step expected by create_target_variable
        rfm["cluster"] = [0, 1]

        df_with_target = create_target_variable(df, rfm)
        self.assertIn("is_high_risk", df_with_target.columns)
        self.assertTrue(df_with_target["is_high_risk"].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
