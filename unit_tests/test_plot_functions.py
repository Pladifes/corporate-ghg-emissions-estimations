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
        plot_path = "data/unit_test_data/results/plots/"
        sector_colors = {"Bucket1": "red", "Bucket2": "blue", "Bucket3": "green"}

        plot_revenue_bucket(df, target, plot_path, sector_colors)
        plot_file = plot_path + f"revenue_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)


class TestPlotRegion(unittest.TestCase):
    def test_plot_generation(self):
        data = {"region": ["Region1", "Region2", "Region3"], "rmses": [0.5, 0.6, 0.7]}
        df = pd.DataFrame(data)
        target = "Test Target"
        plot_path = "data/unit_test_data/results/plots/"
        region_colors = {"Region1": "red", "Region2": "blue", "Region3": "green"}

        plot_region(df, target, plot_path, region_colors)
        plot_file = plot_path + f"region_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)


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
        plot_path = "data/unit_test_data/results/plots/"
        country_colors = {
            "Country1": "red",
            "Country2": "blue",
            "Country3": "green",
            "Country4": "purple",
            "Country5": "orange",
        }

        plot_country(df, target, plot_path, country_colors)

        plot_file = plot_path + f"country_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)


class TestPlotIndustry(unittest.TestCase):
    def test_plot_generation(self):
        data = {
            "industry": ["Industry1", "Industry2", "Industry3"],
            "rmses": [0.5, 0.6, 0.7],
        }
        df = pd.DataFrame(data)
        target = "Test Target"
        plot_path = "data/unit_test_data/results/plots/"
        industry_colors = {
            "Industry1": "red",
            "Industry2": "blue",
            "Industry3": "green",
        }

        plot_industry(df, target, plot_path, industry_colors)
        plot_file = plot_path + f"industry_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)


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
        plot_path = "data/unit_test_data/results/plots/"
        sub_sector_colors = {
            "SubSector1": "red",
            "SubSector2": "blue",
            "SubSector3": "green",
            "SubSector4": "purple",
            "SubSector5": "orange",
        }
        plot_sub_sector(df, target, plot_path, sub_sector_colors)
        plot_file = plot_path + f"sector_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)


class TestPlotYear(unittest.TestCase):
    def test_plot_generation(self):
        data = {"year": [2020, 2021, 2022], "rmses": [0.5, 0.6, 0.7]}
        df = pd.DataFrame(data)

        target = "Test Target"
        plot_path = "data/unit_test_data/results/plots/"
        year_colors = {2020: "red", 2021: "blue", 2022: "green"}

        plot_year(df, target, plot_path, year_colors)

        plot_file = plot_path + f"years_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)


class TestPlotEnergyConsumedLog(unittest.TestCase):
    def test_plot_generation(self):
        data = {"energy_consumed": ["Low", "Medium", "High"], "rmses": [0.5, 0.6, 0.7]}
        df = pd.DataFrame(data)
        target = "Test Target"
        plot_path = "data/unit_test_data/results/plots/"
        energy_colors = {"Low": "red", "Medium": "blue", "High": "green"}

        plot_energy_consumed_log(df, target, plot_path, energy_colors)
        plot_file = plot_path + f"consume_box_plot_{target}.png"
        self.assertTrue(os.path.exists(plot_file))
        os.remove(plot_file)


class TestPlotFunction(unittest.TestCase):
    def test_plot_generation(self):
        X = pd.read_csv("data/unit_test_data/X_train_test.csv")
        y = pd.read_csv("data/unit_test_data/y_train_test.csv").values.reshape(-1)

        model = pickle.load(open("data/unit_test_data/models/cf1_log_model.pkl", "rb"))
        plot_path = "data/unit_test_data/results/plots/"
        target = "Test_Target"
        y_pred = model.predict(X)

        plot(model, X, y, y_pred, plot_path, target)

        shap_plot_file = plot_path + f"shap_{target}.png"
        y_test_y_pred_plot_file = plot_path + f"y_test_y_pred_{target}.png"
        residuals_plot_file = plot_path + f"residus_{target}.png"

        self.assertTrue(os.path.exists(shap_plot_file))
        self.assertTrue(os.path.exists(y_test_y_pred_plot_file))
        self.assertTrue(os.path.exists(residuals_plot_file))

        os.remove(shap_plot_file)
        os.remove(y_test_y_pred_plot_file)
        os.remove(residuals_plot_file)


if __name__ == "__main__":
    unittest.main()
