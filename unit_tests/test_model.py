import unittest
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import os
import sys

app_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(app_folder_path)
from functions.models import catboost_model, xgboost_model, lgbm_model


class TestCatBoostModel(unittest.TestCase):
    def test_catboost_model(self):
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.rand(100, 3), columns=["feature1", "feature2", "feature3"]
        )
        y_train = np.random.rand(100)

        model = catboost_model(X_train, y_train, cross_val=False, verbose=1)

        self.assertIsInstance(model, CatBoostRegressor)


class TestXGBoostModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.rand(100, 3), columns=["feature1", "feature2", "feature3"]
        )
        y_train = pd.Series(np.random.rand(100))

        self.X_train = X_train
        self.y_train = y_train
        model = xgboost_model(self.X_train, self.y_train, cross_val=False, verbose=1)
        self.assertIsInstance(model, XGBRegressor)


class TestLGBMModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.rand(100, 3), columns=["feature1", "feature2", "feature3"]
        )
        y_train = pd.Series(np.random.rand(100))

        self.X_train = X_train
        self.y_train = y_train
        model = lgbm_model(self.X_train, self.y_train)
        self.assertIsInstance(model, LGBMRegressor)


if __name__ == "__main__":
    unittest.main()
