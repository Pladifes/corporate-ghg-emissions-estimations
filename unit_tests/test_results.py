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
        path_rawdata = "data/raw_data/"
        path_benchmark = "benchmark/"
        path_intermediary = "data/intermediary_data/"
        target = "cf1_log"
        extended_features = True
        restricted_features = False
        selec_sect = ["gics_ind", "gics_group"]

        df = pd.read_parquet(path_rawdata + "cgee_preprocessed_dataset_2023.parquet")
        dataset = df[:10000]

        X_train, y_train, X_test, y_test, df_test = custom_train_split(
            dataset,
            path_benchmark,
            path_intermediary,
            target,
            extended_features,
            restricted_features,
            selec_sect,
        )
        path_plot = "plots/"

        result = summary_detailed(
            X_test, y_test, y_pred, df_test, target, restricted_features, path_plot
        )
        y_pred_from_results = pd.read_csv("Estimated_scopes.csv")
        y_pred = y_pred_from_results["cf1_log_e"]
        expected_category_names = []
        actual_category_names = result["category_name"].unique()
        for name in expected_category_names:
            self.assertIn(name, actual_category_names)

        self.assertIsInstance(result, pd.DataFrame)


class TestMetricsFunction(unittest.TestCase):
    def test_metrics(self):
        y_test = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        summary_final = []
        target = "target"
        model_name = "model"

        summary_global, rmse, std = metrics(
            y_test, y_pred, summary_final, target, model_name
        )

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


# class TestBestModelAnalysis(unittest.TestCase):
#     def test_best_model_analysis(self):
#         X_test = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
#         X_train = pd.DataFrame({"feature1": [4, 5, 6], "feature2": [7, 8, 9]})
#         y_test = pd.Series([10, 11, 12])
#         df_test = pd.DataFrame(
#             {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [10, 11, 12]}
#         )
#         target = "target"
#         path_plot =
#         dataset =
#         path_intermediary =
#         summary_metrics_detailed = pd.DataFrame(columns=["Model", "Metric", "Value"])
#         estimated_scopes = pd.DataFrame(columns=["Model", "Scope"])
#         restricted_features = ["feature1"]
#         path_models =

#         (
#             summary_metrics_detailed_updated,
#             estimated_scopes_updated,
#             additional_results,
#         ) = best_model_analysis(
#             None,
#             X_test,
#             X_train,
#             y_test,
#             df_test,
#             target,
#             path_plot,
#             dataset,
#             path_intermediary,
#             summary_metrics_detailed,
#             estimated_scopes,
#             restricted_features,
#             path_models,
#         )

#         self.assertFalse(summary_metrics_detailed_updated.empty)
#         self.assertFalse(estimated_scopes_updated.empty)
#         self.assertIsInstance(additional_results, list)


if __name__ == "__main__":
    unittest.main()
