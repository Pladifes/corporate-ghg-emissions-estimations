import pandas as pd
import numpy as np
import pytest
import sys
import os
import unittest
import pickle

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.plot_functions import (
    plot_revenue_bucket,
    plot_region,
    plot_energy_consumed_log,
    plot_country,
    plot_industry,
    plot_sub_sector,
    plot_year,
    plot,
)


class TestPlotRevenueBucket(unittest.TestCase):
    def test_plot_generation(self):
        data = {
            "revenue_bucket": ["Bucket1", "Bucket2", "Bucket3"],
            "rmses": [0.5, 0.6, 0.7],
        }
        df = pd.DataFrame(data)
        target = "Test Target"
        plot_path = "unit_tests/test_plots/"
        sector_colors = {"Bucket1": "red", "Bucket2": "blue", "Bucket3": "green"}
        os.makedirs(plot_path, exist_ok=True)
        plot_revenue_bucket(df, target, plot_path, sector_colors)
        plot_file = plot_path + f"revenue_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)
        os.rmdir(plot_path)


class TestPlotRegion(unittest.TestCase):
    def test_plot_generation(self):
        data = {"region": ["Region1", "Region2", "Region3"], "rmses": [0.5, 0.6, 0.7]}
        df = pd.DataFrame(data)
        target = "Test Target"
        plot_path = "unit_tests/test_plots/"
        region_colors = {"Region1": "red", "Region2": "blue", "Region3": "green"}
        os.makedirs(plot_path, exist_ok=True)
        plot_region(df, target, plot_path, region_colors)
        plot_file = plot_path + f"region_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)
        os.rmdir(plot_path)


class TestPlotCountry(unittest.TestCase):
    def test_plot_generation(self):
        data = {
            "country": [
                "Country1",
                "Country2",
                "Country3",
                "Country4",
                "Country5",
            ],
            "rmses": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
        df = pd.DataFrame(data)
        target = "Test Target"
        plot_path = "unit_tests/test_plots/"
        country_colors = {
            "Country1": "red",
            "Country2": "blue",
            "Country3": "green",
            "Country4": "purple",
            "Country5": "orange",
        }

        os.makedirs(plot_path, exist_ok=True)

        plot_country(df, target, plot_path, country_colors)

        plot_file = plot_path + f"country_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)
        os.rmdir(plot_path)


class TestPlotIndustry(unittest.TestCase):
    def test_plot_generation(self):
        data = {
            "industry": ["Industry1", "Industry2", "Industry3"],
            "rmses": [0.5, 0.6, 0.7],
        }
        df = pd.DataFrame(data)
        target = "Test Target"
        plot_path = "unit_tests/test_plots/"
        industry_colors = {
            "Industry1": "red",
            "Industry2": "blue",
            "Industry3": "green",
        }
        os.makedirs(plot_path, exist_ok=True)
        plot_industry(df, target, plot_path, industry_colors)
        plot_file = plot_path + f"industry_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)
        os.rmdir(plot_path)


class TestPlotSubSector(unittest.TestCase):
    def test_plot_generation(self):
        data = {
            "sub_sector": [
                "SubSector1",
                "SubSector2",
                "SubSector3",
                "SubSector4",
                "SubSector5",
            ],
            "rmses": [0.5, 0.6, 0.7, 0.8, 0.2],
        }
        df = pd.DataFrame(data)

        target = "Test Target"
        plot_path = "unit_tests/test_plots/"
        sub_sector_colors = {
            "SubSector1": "red",
            "SubSector2": "blue",
            "SubSector3": "green",
            "SubSector4": "purple",
            "SubSector5": "orange",
        }

        os.makedirs(plot_path, exist_ok=True)

        plot_sub_sector(df, target, plot_path, sub_sector_colors)

        plot_file = plot_path + f"sector_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)
        os.rmdir(plot_path)


class TestPlotYear(unittest.TestCase):
    def test_plot_generation(self):
        data = {"year": [2020, 2021, 2022], "rmses": [0.5, 0.6, 0.7]}
        df = pd.DataFrame(data)

        target = "Test Target"
        plot_path = "unit_tests/test_plots/"
        year_colors = {2020: "red", 2021: "blue", 2022: "green"}

        os.makedirs(plot_path, exist_ok=True)

        plot_year(df, target, plot_path, year_colors)

        plot_file = plot_path + f"years_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)
        os.rmdir(plot_path)


class TestPlotEnergyConsumedLog(unittest.TestCase):
    def test_plot_generation(self):
        data = {"energy_consumed": ["Low", "Medium", "High"], "rmses": [0.5, 0.6, 0.7]}
        df = pd.DataFrame(data)
        target = "Test Target"
        plot_path = "unit_tests/test_plots/"
        energy_colors = {"Low": "red", "Medium": "blue", "High": "green"}
        os.makedirs(plot_path, exist_ok=True)
        plot_energy_consumed_log(df, target, plot_path, energy_colors)
        plot_file = plot_path + f"consume_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)
        os.rmdir(plot_path)


class TestPlotFunction(unittest.TestCase):
    def test_plot_generation(self):
        X = pd.read_parquet("data/intermediary_data/df_train.parquet")
        y_test = X["cf1_log"]
        y_pred = X["cf1_log"]

        model = pickle.load("models/cf1_log_model.pkl")
        plot_path = "test_plots/"
        target = "Test_Target"

        os.makedirs(plot_path, exist_ok=True)
        plot(model, X, y_test, y_pred, plot_path, target)

        shap_plot_file = plot_path + f"shap_{target}.png"
        y_test_y_pred_plot_file = plot_path + f"y_test_y_pred_{target}.png"
        residuals_plot_file = plot_path + f"residus_{target}.png"

        self.assertTrue(os.path.exists(shap_plot_file))
        self.assertTrue(os.path.exists(y_test_y_pred_plot_file))
        self.assertTrue(os.path.exists(residuals_plot_file))

        os.remove(shap_plot_file)
        os.remove(y_test_y_pred_plot_file)
        os.remove(residuals_plot_file)
        os.rmdir(plot_path)


if __name__ == "__main__":
    unittest.main()
