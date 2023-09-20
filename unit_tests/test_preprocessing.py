import pandas as pd
import numpy as np
import unittest
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.preprocessing import (
    target_preprocessing,
    df_split,
    set_columns,
    logtransform,
    label_final_co2_law,
    processingrawdata,
    outliers_preprocess,
    selected_features,
    custom_train_split,
)


class TestTargetPreprocessing(unittest.TestCase):
    def test_cf1_log(self):
        path_dataset_loading = "data/intermediary_data/"
        df = pd.read_parquet(path_dataset_loading + "df_train.parquet")
        result = target_preprocessing(df, "cf1_log")

        self.assertTrue(all(result["fiscal_year"] >= 2005))
        self.assertTrue("cf1_log" in result.columns)
        self.assertTrue(result["cf1_log"].isna().sum() == 0)

    def test_cf2_log(self):
        path_dataset_loading = "data/intermediary_data/"
        df = pd.read_parquet(path_dataset_loading + "df_train.parquet")
        result = target_preprocessing(df, "cf2_log")

        self.assertTrue(all(result["fiscal_year"] >= 2005))
        self.assertTrue("cf2_log" in result.columns)
        self.assertTrue(result["cf2_log"].isna().sum() == 0)


class TestDfSplit(unittest.TestCase):
    def setUp(self):

        raw_data_path = "data/raw_data/"
        self.test_df = pd.read_parquet(
            raw_data_path + "cgee_preprocessed_dataset_2023.parquet"
        )

    def test_df_split(self):
        benchmark_df = pd.read_csv("benchmark/lst_companies_test_GICS_2023.csv")
        df_train, df_test = df_split(self.test_df, "benchmark/")

        expected_train = self.test_df[
            ~self.test_df["company_name"].isin(benchmark_df["company_name"])
        ]
        expected_test = self.test_df[
            self.test_df["company_name"].isin(benchmark_df["company_name"])
        ]

        self.assertTrue(df_train.equals(expected_train))
        self.assertTrue(df_test.equals(expected_test))


class TestSetColumns(unittest.TestCase):
    def setUp(self):
        data = {
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6],
        }
        self.test_df = pd.DataFrame(data)

    def test_set_columns(self):
        features = ["feature_1", "feature_2", "feature_3"]
        result_df = set_columns(self.test_df, features)
        self.assertTrue("feature_3" in result_df.columns)
        self.assertTrue(all(result_df["feature_3"] == 0))

    def test_set_columns_no_missing(self):
        features = ["feature_1", "feature_2"]
        result_df = set_columns(self.test_df, features)
        self.assertTrue(self.test_df.equals(result_df))


class TestLogTransform(unittest.TestCase):
    def setUp(self):
        data = {
            "cf1": [1, 2, 3],
            "cf2": [4, 5, 6],
            "cf3": [7, 8, 9],
            "cf123": [10, 11, 12],
        }
        self.test_df = pd.DataFrame(data)

    def test_logtransform_train(self):
        result_df = logtransform(
            self.test_df,
            ["cf1", "cf2", "cf3", "cf123"],
            "unit_tests/test_dataset/",
            train=True,
        )

        self.assertTrue("cf1_log" in result_df.columns)
        self.assertTrue(all(result_df["cf1_log"] == np.log10(self.test_df["cf1"] + 1)))
        self.assertTrue("cf2_log" in result_df.columns)
        self.assertTrue("cf3_log" in result_df.columns)
        self.assertTrue(all(result_df["cf3_log"] == np.log10(self.test_df["cf3"] + 1)))
        self.assertTrue("cf123_log" in result_df.columns)
        self.assertTrue(
            all(result_df["cf123_log"] == np.log10(self.test_df["cf123"] + 1))
        )

        self.assertTrue(os.path.isfile("columns_min.csv"))

    def test_logtransform_test(self):
        columns_min_data = {
            "column": ["cf1", "cf2", "cf3", "cf123"],
            "min_value": [0, 1, 2, 3],
        }
        columns_min_df = pd.DataFrame(columns_min_data)
        columns_min_df.to_csv("columns_min.csv", index=False)

        result_df = logtransform(
            self.test_df, ["cf1", "cf2", "cf3", "cf123"], "", train=False
        )

        self.assertTrue("cf1_log" in result_df.columns)
        self.assertTrue(
            all(result_df["cf1_log"] == np.log10(self.test_df["cf1"] - 0 + 1))
        )
        self.assertTrue("cf2_log" in result_df.columns)
        self.assertTrue("cf3_log" in result_df.columns)

        self.assertTrue("cf123_log" in result_df.columns)

        if os.path.isfile("columns_min.csv"):
            os.remove("columns_min.csv")


class TestLabelFinalCO2Law(unittest.TestCase):
    def test_label_final_co2_law_impl(self):
        row = pd.Series({"co2_law": "Yes", "co2_status": "Implemented"})
        result = label_final_co2_law(row)
        self.assertEqual(result, "has_implemented_CO2Law")

    def test_label_final_co2_law_not_impl(self):
        row = pd.Series({"co2_law": "Yes", "co2_status": "Not Implemented"})
        result = label_final_co2_law(row)
        self.assertEqual(result, "has_not_implemented_CO2Law")

    def test_label_final_co2_law_not_yes(self):
        row = pd.Series({"co2_law": "No", "co2_status": "Implemented"})
        result = label_final_co2_law(row)
        self.assertEqual(result, "has_not_implemented_CO2Law")


class TestProcessingRawData(unittest.TestCase):
    def setUp(self):
        data = {
            "gics_group": ["Group1", "Group2", "Group3"],
            "gics_sector": ["Sector1", "Sector2", "Sector3"],
            "gics_ind": ["Ind1", "Ind2", "Ind3"],
            "gics_sub_ind": ["SubInd1", "SubInd2", "SubInd3"],
            "income_group": ["H", "UM", None],
            "co2_law": ["Yes", "No", "Yes"],
            "co2_status": ["Yes", "No", "Yes"],
        }
        self.test_df = pd.DataFrame(data)

    def test_processingrawdata(self):
        result_df = processingrawdata(
            self.test_df, restricted_features=True, train=True
        )

        self.assertTrue("final_co2_law_encoded" in result_df.columns)

        self.assertTrue("income_group_encoded" in result_df.columns)

    def test_processingrawdata_full(self):

        result_df = processingrawdata(
            self.test_df, restricted_features=False, train=True
        )

        self.assertTrue("gics_group_Group1" in result_df.columns)
        self.assertTrue("gics_sector_Sector1" in result_df.columns)


class TestSelectedFeatures(unittest.TestCase):
    def setUp(self):
        data_train = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
        }
        data_test = {
            "feature1": [7, 8, 9],
            "feature2": [10, 11, 12],
        }
        self.df_train = pd.DataFrame(data_train)
        self.df_test = pd.DataFrame(data_test)

    def test_selected_features(self):
        result_features = selected_features(
            self.df_train,
            self.df_test,
            "",
            ["extended_feature1", "extended_feature2"],
            ["sect1", "sect2"],
        )
        self.assertTrue(os.path.isfile("features.csv"))

    def tearDown(self):
        if os.path.isfile("features.csv"):
            os.remove("features.csv")


class TestOutliersPreprocess(unittest.TestCase):
    def test_outliers_preprocess(self):
        path_intermediary = "data/intermediary_data/"
        df = pd.read_parquet(path_intermediary + "df_train.parquet")

        processed_df = outliers_preprocess(df, "cf1_log")

        expected_columns = "intensity_cf1_log"
        self.assertIn(expected_columns, list(processed_df.columns))

        for subindustry in processed_df["gics_sub_ind"].unique():
            subset = processed_df[processed_df["gics_sub_ind"] == subindustry]
            if len(subset) > 10:
                median_subind = subset["intensity_cf1_log"].quantile(0.50)
                Q1 = subset["intensity_cf1_log"].quantile(0.25)
                Q3 = subset["intensity_cf1_log"].quantile(0.75)
                IQR = Q3 - Q1

                max_subind = median_subind + 2.5 * IQR
                min_subind = median_subind - 1.5 * IQR

                self.assertFalse(
                    all(
                        (subset["intensity_cf1_log"] >= min_subind)
                        & (subset["intensity_cf1_log"] <= max_subind)
                    )
                )


class TestCustomTrainSplit(unittest.TestCase):
    def test_custom_train_split(self):
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
        features_df = pd.read_csv(path_intermediary + "features.csv")
        featureslst = list(features_df.features.unique())

        self.assertListEqual(list(X_train.columns), featureslst)
        self.assertListEqual(list(X_test.columns), featureslst)


if __name__ == "__main__":
    unittest.main()
