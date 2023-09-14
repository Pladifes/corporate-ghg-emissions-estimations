import unittest
import pandas as pd
import sys
import os

app_folder_path = os.path.abspath(
os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(app_folder_path)
from functions.loading import load_data


class TestDataLoading(unittest.TestCase):
    def test_load_data_existing_file(self):
        temp_df = pd.read_csv("unit_tests/test_dataset/test_dataset.csv")
        temp_df.to_parquet("unit_tests/test_dataset/cgee_preprocessed_dataset_2023.parquet")

        loaded_data = load_data(
            "unit_tests/test_dataset/",
            filter_outliers=True,
            threshold_under=1.5,
            threshold_over=2.5,
            save=False,
        )

        self.assertIsInstance(loaded_data, pd.DataFrame)

        expected_columns = list(temp_df.columns)
        self.assertListEqual(list(loaded_data.columns), expected_columns)

        os.remove("unit_tests/test_dataset/cgee_preprocessed_dataset_2023.parquet")

if __name__ == "__main__":
    unittest.main()
