import unittest
import pandas as pd
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.apply_model import apply_model_on_forbes_data


class TestApplyModelOnForbesData(unittest.TestCase):
    def setUp(self):
        self.path_rawdata = "data/raw_data/"
        self.path_results = "results/"
        self.path_intermediary = "data/intermediary_data/"
        self.path_models = "models/"

        self.sample_forbes_data = pd.read_excel(
            self.path_rawdata + "forbes_2007_2022_completed.xlsx"
        )

    def test_apply_model_on_forbes_data(self):
        result_df = apply_model_on_forbes_data(
            path_rawdata=self.path_rawdata,
            path_results=self.path_results,
            path_intermediary=self.path_intermediary,
            path_models=self.path_models,
            save=True,
        )
        self.assertIsNotNone(result_df)
        self.assertTrue("cf1_e" in result_df.columns)
        self.assertTrue("cf2_e" in result_df.columns)
        self.assertTrue("cf3_e" in result_df.columns)
        self.assertTrue("cf123_e" in result_df.columns)
        self.assertTrue("cf1_e + cf2_e + cf3_e" in result_df.columns)

        self.assertTrue(
            os.path.isfile(self.path_results + "pladifes_free_emissions_estimates.xlsx")
        )

if __name__ == "__main__":
    unittest.main()
