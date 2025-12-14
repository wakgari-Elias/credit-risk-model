import unittest
import pandas as pd
from src.data_processing import (
    load_csv,
    numeric_summary_stats,
    categorical_summary,
    task3_feature_engineering_pipeline
)

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # ---------- Task-2 sample dataframe ----------
        self.df = pd.DataFrame({
            "Amount": [10, 20, 30],
            "Value": [100, 200, 300],
            "Category": ["A", "B", "A"]
        })

        # ---------- Task-3 sample dataframe ----------
        self.df_task3 = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3, 3],
            'TransactionId': [101, 102, 201, 202, 301, 302],
            'TransactionStartTime': [
                '2025-12-10 08:00:00', '2025-12-12 12:00:00',
                '2025-12-11 09:30:00', '2025-12-13 14:45:00',
                '2025-12-12 10:15:00', '2025-12-14 16:30:00'
            ],
            'Amount': [100, 150, 200, 250, 300, 350],
            'Category': ['A', 'B', 'A', 'B', 'A', 'B']
        })

        # Fix for Task-3 pipeline
        # After RFM and temporal features, these columns exist
         # Fix: only include original columns
        self.numeric_features = ['Amount']
        self.categorical_features = ['Category']
        self.wo_features = None


    # ---------- Task-2 Tests ----------
    def test_numeric_summary_stats(self):
        result = numeric_summary_stats(self.df)
        self.assertIn("skew", result.columns)
        self.assertIn("kurtosis", result.columns)
        self.assertEqual(result.shape[0], 2)

    def test_categorical_summary(self):
        result = categorical_summary(self.df)
        self.assertIn("Category", result)
        self.assertEqual(result["Category"]["top"], "A")

    def test_load_csv_nonexistent(self):
        result = load_csv("nonexistent.csv")
        self.assertIsNone(result)

    # ---------- Task-3 Tests ----------
    def test_task3_pipeline_runs(self):
        df_transformed = task3_feature_engineering_pipeline(
            self.df_task3,
            self.numeric_features,
            self.categorical_features,
            self.wo_features
        )
        # Test returns DataFrame
        self.assertIsInstance(df_transformed, pd.DataFrame)
        # Test contains new RFM columns
        expected_cols = ['Recency','Frequency','MonetarySum','MonetaryAvg','MonetaryStd']
        for col in expected_cols:
            self.assertIn(col, df_transformed.columns)

if __name__ == "__main__":
    unittest.main()
