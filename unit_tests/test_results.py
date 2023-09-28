import sys
import os
import unittest
import numpy as np
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.results import (
    gics_to_name,
    summary_detailed,
    metrics,
    best_model_analysis,
)
from functions.preprocessing import custom_train_split


class TestGICSToName(unittest.TestCase):
    def test_gics_to_name(self):
        self.assertEqual(gics_to_name(10.0), "Energy")
        self.assertEqual(gics_to_name(15.0), "Materials")
        self.assertEqual(gics_to_name(20.0), "Industrials")
        self.assertEqual(gics_to_name(25.0), "Cons. Discretionary")
        self.assertEqual(gics_to_name(30.0), "Cons. Staples")
        self.assertEqual(gics_to_name(35.0), "Health Care")
        self.assertEqual(gics_to_name(40.0), "Financials")
        self.assertEqual(gics_to_name(45.0), "IT")
        self.assertEqual(gics_to_name(50.0), "Telecommunication")
        self.assertEqual(gics_to_name(55.0), "Utilities")
        self.assertEqual(gics_to_name(60.0), "Real Estate")
        self.assertEqual(gics_to_name(99.0), 99.0)


class TestSummaryDetailed(unittest.TestCase):
    def test_summary_detailed(self):
        path_test_data = "data/unit_test_data/"
        X = pd.read_csv(path_test_data + "X_train_test.csv")
        X["revenue"] = 10**X.revenue_log
        X["gics_name"] = ["sector_1" for i in range(len(X))]
        X["region"] = ["region_1" for i in range(len(X))]
        X["country_hq"] = ["country_1" for i in range(len(X))]
        y = pd.read_csv(path_test_data + "y_train_test.csv")
        target = "cf1_log"
        restricted_features = True
        path_plot = path_test_data + "results/plots/"

        result = summary_detailed(X, y, y, X, target, restricted_features, path_plot)
        self.assertIsInstance(result, pd.DataFrame)


class TestMetricsFunction(unittest.TestCase):
    def test_metrics(self):
        y_test = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 1.1, 2.2, 3.3, 4.4, 5.5])
        summary_final = []
        target = "target"
        model_name = "model"

        summary_global, rmse, std = metrics(y_test, y_pred, summary_final, target, model_name)

        self.assertIsInstance(summary_global, pd.DataFrame)

        expected_columns = [
            "Target",
            "model",
            "mae",
            "mse",
            "r2",
            "rmse",
            "mape",
            "std",
        ]
        self.assertListEqual(list(summary_global.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()
