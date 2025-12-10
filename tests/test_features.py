"""Tests for feature engineering."""
import unittest
import pandas as pd
import numpy as np
from src.feature_engineer import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def test_interactions(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        fe = FeatureEngineer()
        result = fe.create_interactions(df, ["a", "b"])
        self.assertIn("a_x_b", result.columns)

    def test_statistical_features(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        fe = FeatureEngineer()
        result = fe.create_statistical_features(df, ["a", "b", "c"])
        self.assertIn("row_mean", result.columns)

if __name__ == "__main__":
    unittest.main()
