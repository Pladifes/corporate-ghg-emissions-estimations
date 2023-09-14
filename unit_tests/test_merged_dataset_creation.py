import unittest
import pandas as pd
import sys
import os

app_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(app_folder_path)
from functions.merged_dataset_creation import merge_datasets


class TestMergeDatasets(unittest.TestCase):
    def setUp(self):
        raw_data_path = "data/raw_data/"
        self.input_dataset = pd.read_csv("unit_tests/test_dataset/test_dataset.csv")
        self.carbon_pricing = pd.read_csv(
            raw_data_path + "carbon_pricing_preprocessed_2023.csv"
        )
        self.income_group = pd.read_csv(
            raw_data_path + "income_group_preprocessed_2023.csv"
        )
        self.fuel_intensity = pd.read_csv(raw_data_path + "fuel_mix_2023.csv")
        self.region_mapping = pd.read_excel(
            raw_data_path + "country_region_mapping.xlsx"
        )

    def test_merge_datasets(self):

        merged_df = merge_datasets(
            self.input_dataset,
            self.carbon_pricing,
            self.income_group,
            self.fuel_intensity,
            self.region_mapping,
        )

        self.assertTrue("co2_law" in list(merged_df.columns))
        self.assertTrue("income_group" in merged_df.columns)
        self.assertTrue("fuel_intensity" in merged_df.columns)
        self.assertTrue("region" in merged_df.columns)


if __name__ == "__main__":
    unittest.main()
