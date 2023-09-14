import unittest
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os
import sys

app_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(app_folder_path)
from functions.models import catboost_model


class TestCatBoostModel(unittest.TestCase):
    def test_catboost_model(self):
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.rand(100, 3), columns=["feature1", "feature2", "feature3"]
        )
        y_train = np.random.rand(100)

        model = catboost_model(X_train, y_train)

        self.assertIsInstance(model, CatBoostRegressor)

        # model_cv = catboost_model(X_train, y_train, cross_val=True)
        # self.assertIsInstance(model_cv, CatBoostRegressor)


if __name__ == "__main__":
    unittest.main()
