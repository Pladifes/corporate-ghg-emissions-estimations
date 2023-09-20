import unittest
import pandas as pd
import sys
import os

app_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(app_folder_path)
from functions.loading import load_data


class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_parquet(
            "data/raw_data/cgee_preprocessed_dataset_2023.parquet"
        )
        self.test_data = self.data[:1000]
        self.temp_file_path = (
            "unit_tests/test_dataset/cgee_preprocessed_dataset_2023.parquet"
        )

    def tearDown(self):
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def test_data_loading(self):
        self.assertIsInstance(self.test_data, pd.DataFrame)

    def test_load_data_existing_file(self):
        self.test_data.to_parquet(self.temp_file_path)

        loaded_data = load_data(
            "unit_tests/test_dataset/",
            filter_outliers=True,
            threshold_under=1.5,
            threshold_over=2.5,
            save=False,
        )

        self.assertIsInstance(loaded_data, pd.DataFrame)

        expected_columns = list(self.test_data.columns)
        self.assertListEqual(list(loaded_data.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()
