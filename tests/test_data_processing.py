# tests/test_data_processing.py
import unittest
import pandas as pd
from src.data_processing import load_csv, numeric_summary_stats, categorical_summary

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "Amount": [10, 20, 30],
            "Value": [100, 200, 300],
            "Category": ["A", "B", "A"]
        })

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

if __name__ == "__main__":
    unittest.main()
