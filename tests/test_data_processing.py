import unittest
import pandas as pd
import numpy as np

from src.data_processing import (
    load_csv,
    numeric_summary_stats,
    categorical_summary,
    missing_values_report,
    build_task3_pipeline,
)


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # -----------------------------
        # Task-2 Sample Data
        # -----------------------------
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

    def test_load_csv_nonexistent(self):
        """load_csv should return None for missing file."""
        result = load_csv("non_existent_file.csv")
        self.assertIsNone(result)

    def test_numeric_summary_stats(self):
        """Numeric summary should include skew and kurtosis."""
        result = numeric_summary_stats(self.df_task2)
        self.assertIn("skew", result.columns)
        self.assertIn("kurtosis", result.columns)
        self.assertEqual(result.shape[0], 2)  # Amount, Value

    def test_categorical_summary(self):
        """Categorical summary should detect top category."""
        result = categorical_summary(self.df_task2)
        self.assertIn("Category", result)
        self.assertEqual(result["Category"]["top"], "A")

    def test_missing_values_report(self):
        """Missing values report should detect NaNs."""
        df = self.df_task2.copy()
        df.loc[0, "Amount"] = np.nan
        report = missing_values_report(df)
        self.assertIn("Amount", report.index)

    # =====================================================
    # Task-3 Tests (Feature Engineering Pipeline)
    # =====================================================

    def test_task3_pipeline_runs(self):
        """Pipeline should run end-to-end without errors."""
        pipeline = build_task3_pipeline()
        X_transformed = pipeline.fit_transform(self.df_task3)

        self.assertIsNotNone(X_transformed)
        self.assertTrue(hasattr(X_transformed, "shape"))

    def test_rfm_feature_dimensions(self):
        """Pipeline output rows should match unique customers."""
        pipeline = build_task3_pipeline()
        X_transformed = pipeline.fit_transform(self.df_task3)

        n_customers = self.df_task3["CustomerId"].nunique()
        self.assertEqual(X_transformed.shape[0], n_customers)

    def test_pipeline_output_is_numeric(self):
        """Final output should be fully numeric and model-ready."""
        pipeline = build_task3_pipeline()
        X_transformed = pipeline.fit_transform(self.df_task3)

        self.assertTrue(np.issubdtype(X_transformed.dtype, np.number))


if __name__ == "__main__":
    unittest.main()
