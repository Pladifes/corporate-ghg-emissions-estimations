import unittest
import pandas as pd
import sys
import os

app_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(app_folder_path)
from functions.loading import load_data


class TestDataLoading(unittest.TestCase):
    def test_load_data_raw_existing_file(self):
        loaded_data = load_data(
            "data/unit_test_data/",
            filter_outliers=True,
            threshold_under=1.5,
            threshold_over=2.5,
            save=False,
        )

        self.assertIsInstance(loaded_data, pd.DataFrame)

        expected_columns = [
            "company_id",
            "company_name",
            "isin",
            "ticker",
            "country_hq",
            "gics_sector",
            "gics_group",
            "gics_ind",
            "gics_sub_ind",
            "gics_name",
            "fiscal_year",
            "revenue",
            "ebitda",
            "ebit",
            "capex",
            "gppe",
            "nppe",
            "accu_dep",
            "intan",
            "cogs",
            "gmar",
            "asset",
            "lt_debt",
            "employees",
            "energy_produced",
            "energy_consumed",
            "cf1",
            "cf2",
            "cf3",
            "cf123",
            "co2_law",
            "co2_scheme",
            "co2_status",
            "co2_coverage",
            "start_year",
            "status",
            "price",
            "area",
            "year",
            "fuel_intensity",
            "income_group",
            "region",
            "age",
            "cap_inten",
            "leverage",
            "intensity_cf1",
            "intensity_cf2",
            "intensity_cf3",
            "intensity_cf123",
        ]
        self.assertListEqual(list(loaded_data.columns), expected_columns)

    def test_load_data_preprocessed_existing_file(self):
        loaded_data = load_data(
            "data/unit_test_data/preprocessed/",
            filter_outliers=True,
            threshold_under=1.5,
            threshold_over=2.5,
            save=False,
        )

        self.assertIsInstance(loaded_data, pd.DataFrame)

        expected_columns = [
            "company_id",
            "company_name",
            "isin",
            "ticker",
            "country_hq",
            "gics_sector",
            "gics_group",
            "gics_ind",
            "gics_sub_ind",
            "gics_name",
            "fiscal_year",
            "revenue",
            "ebitda",
            "ebit",
            "capex",
            "gppe",
            "nppe",
            "accu_dep",
            "intan",
            "cogs",
            "gmar",
            "asset",
            "lt_debt",
            "employees",
            "energy_produced",
            "energy_consumed",
            "cf1",
            "cf2",
            "cf3",
            "cf123",
            "co2_law",
            "co2_scheme",
            "co2_status",
            "co2_coverage",
            "start_year",
            "status",
            "price",
            "area",
            "year",
            "fuel_intensity",
            "income_group",
            "region",
            "age",
            "cap_inten",
            "leverage",
            "intensity_cf1",
            "intensity_cf2",
            "intensity_cf3",
            "intensity_cf123",
        ]
        self.assertListEqual(list(loaded_data.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()
