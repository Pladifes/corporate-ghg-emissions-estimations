import pickle
import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from bayes_opt import BayesianOptimization

from functions.preprocessing import encoding
from functions.preprocessing import set_columns


def features_selectkbest(df, target):
    """
    Selects the k best features from a pandas DataFrame using the SelectKBest algorithm
    and returns a list of lists with the feature names.

    Args:
        df: A pandas DataFrame containing the features and the target variable.
        target: The name of the target variable column in the DataFrame.

    Returns:
        A list of lists with the names of the k best features selected by the SelectKBest algorithm.
    """
    all_features = []
    df = df.select_dtypes(include=["float64", "int64", "uint8"])
    keep_cols = [col for col in df.columns if (col.startswith("CF") and col != target)]
    cols_with_no_missing = df[keep_cols].columns[df[keep_cols].isnull().sum() == 0]
    df_final = df[cols_with_no_missing]
    print(df_final.shape)
    for k in range(10, df_final.shape[1], 20):
        selector = SelectKBest(k=k)
        y = df_final[target]
        X = df_final.drop([target], axis=1)
        selector.fit(X, y)
        X_new = selector.transform(X)
        mask = selector.get_support()
        feature_names = X.columns[mask]
        all_features.append(list(feature_names))
    return all_features


def xgboost_model(
    X_train, y_train, cross_val=False, verbose=0, n_jobs=-1, n_iter=10, seed=None, weights=None, custom_gradient=False
):
    """
    This function returns a XGBoost model for regression tasks
    Warning : if cross_val is set True, the function is way longer (several minutes,
    often less than an hour)
    """
    model = XGBRegressor(random_state=seed, verbosity=0)

    if cross_val:

        def xgb_evaluate(n_estimators, learning_rate, max_depth, gamma, subsample, colsample_bytree):
            params = {
                "n_estimators": int(n_estimators),
                "learning_rate": learning_rate,
                "max_depth": int(max_depth),
                "gamma": gamma,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }

            model = XGBRegressor(verbose=verbose, random_state=seed, **params)
            score = cross_val_score(model, X_train, y_train, scoring="r2", cv=3, n_jobs=n_jobs).mean()
            return score

        xgb_bo = BayesianOptimization(
            xgb_evaluate,
            {
                "n_estimators": (10, 300),
                "learning_rate": (0.01, 0.5),
                "max_depth": (1, 100),
                "gamma": (0, 0.1),
                "subsample": (0.5, 1),
                "colsample_bytree": (0.5, 1),
            },
            verbose=0,
            random_state=seed,
        )
        xgb_bo.maximize(init_points=10, n_iter=n_iter, acq="ei", xi=0.0)

        model = XGBRegressor(
            verbose=verbose,
            random_state=seed,
            n_estimators=int(xgb_bo.max["params"]["n_estimators"]),
            learning_rate=xgb_bo.max["params"]["learning_rate"],
            max_depth=int(xgb_bo.max["params"]["max_depth"]),
            gamma=xgb_bo.max["params"]["gamma"],
            subsample=xgb_bo.max["params"]["subsample"],
            colsample_bytree=xgb_bo.max["params"]["colsample_bytree"],
        )

    if custom_gradient:
        print("error, no custom gradient with xgboost, error")

    else:
        if weights is not None:
            model.fit(
                X_train,
                y_train,
                sample_weight=weights,
                verbose=False,
            )
        else:
            model.fit(
                X_train,
                y_train,
                verbose=False,
            )

    return model


def lgbm_model(
    X_train, y_train, cross_val=False, n_jobs=-1, verbose=-1, n_iter=10, seed=None, weights=None, custom_gradient=False
):
    """
    This function returns a LGBM model for regression tasks
    Warning : if cross_val is set True, the function is way longer (several minutes,
    often less than an hour)
    """
    verbose -= 1
    # global companies_lst

    if custom_gradient == "L1":
        # companies_lst = pd.read_csv("data/intermediary_data/companies_ids.csv").squeeze().tolist()
        model = LGBMRegressor(verbose=verbose, random_state=seed, objective=RmseObjectiveL1().calc_ders_range)
    elif custom_gradient == "L2":
        # companies_lst = pd.read_csv("data/intermediary_data/companies_ids.csv").squeeze().tolist()
        model = LGBMRegressor(verbose=verbose, random_state=seed, objective=RmseObjectiveL2().calc_ders_range)
    else:
        model = LGBMRegressor(verbose=verbose, random_state=seed)

    if cross_val:

        def lgbm_evaluate(
            n_estimators,
            learning_rate,
            num_leaves,
            subsample,
            colsample_bytree,
            max_depth,
            num_boost_round,
            reg_alpha,
            min_data_in_leaf,
            lambda_l1,
            lambda_l2,
        ):
            params = {
                "n_estimators": int(n_estimators),
                "learning_rate": learning_rate,
                "num_leaves": int(num_leaves),
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "max_depth": int(max_depth),
                "num_boost_round": int(num_boost_round),
                "reg_alpha": reg_alpha,
                "min_data_in_leaf": int(min_data_in_leaf),
                "lambda_l1": lambda_l1,
                "lambda_l2": int(lambda_l2),
            }

            model = LGBMRegressor(random_state=seed, **params)
            score = cross_val_score(model, X_train, y_train, scoring="r2", cv=3, n_jobs=n_jobs).mean()
            return score

        lgbm_bo = BayesianOptimization(
            lgbm_evaluate,
            {
                "n_estimators": (50, 500),
                "learning_rate": (0.01, 0.5),
                "num_leaves": (31, 100),
                "subsample": (0.5, 1),
                "colsample_bytree": (0.5, 1),
                "max_depth": (1, 20),
                "num_boost_round": (100, 3000),
                "reg_alpha": (0.1, 0.5),
                "min_data_in_leaf": (20, 400),
                "lambda_l1": (0, 1.5),
                "lambda_l2": (0, 1),
            },
            verbose=0,
            random_state=seed,
        )
        lgbm_bo.maximize(init_points=10, n_iter=n_iter, acq="ei", xi=0.0)

        model = LGBMRegressor(
            verbose=verbose,
            random_state=seed,
            n_estimators=int(lgbm_bo.max["params"]["n_estimators"]),
            learning_rate=lgbm_bo.max["params"]["learning_rate"],
            num_leaves=int(lgbm_bo.max["params"]["num_leaves"]),
            subsample=lgbm_bo.max["params"]["subsample"],
            colsample_bytree=lgbm_bo.max["params"]["colsample_bytree"],
            max_depth=int(lgbm_bo.max["params"]["max_depth"]),
        )

    if custom_gradient:
        model.fit(
            X_train,
            y_train,
            eval_metric=rmse_metric_lgbm,
            sample_weight=weights,
            eval_sample_weight=weights,
        )
    else:
        if weights is not None:
            model.fit(
                X_train,
                y_train,
                sample_weight=weights,
                eval_sample_weight=weights,
            )
        else:
            model.fit(
                X_train,
                y_train,
            )

    return model


def catboost_model(
    X_train, y_train, cross_val=False, n_jobs=-1, verbose=0, n_iter=10, seed=None, weights=None, custom_gradient=False
):
    """
    This function returns a CatBoost model
    Warning : if cross_val is set True, the function is way longer (several minutes,
    often less than an hour)
    """
    # global companies_lst

    if custom_gradient == "L1":
        # companies_lst = pd.read_csv("data/intermediary_data/companies_ids.csv").squeeze().tolist()
        model = CatBoostRegressor(
            verbose=verbose, random_state=seed, loss_function=RmseObjectiveL1(), eval_metric=RmseMetric()
        )
    elif custom_gradient == "L2":
        # companies_lst = pd.read_csv("data/intermediary_data/companies_ids.csv").squeeze().tolist()
        model = CatBoostRegressor(
            verbose=verbose, random_state=seed, loss_function=RmseObjectiveL2(), eval_metric=RmseMetric()
        )
    else:
        model = CatBoostRegressor(verbose=verbose, random_state=seed)

    if cross_val:

        def catboost_evaluate(depth, learning_rate, l2_leaf_reg, rsm, subsample, iterations):
            params = {
                "depth": int(depth),
                "learning_rate": learning_rate,
                "l2_leaf_reg": l2_leaf_reg,
                "rsm": rsm,
                "subsample": subsample,
                "iterations": int(iterations),
            }
            model = CatBoostRegressor(verbose=verbose, random_state=seed, **params)
            score = cross_val_score(model, X_train, y_train, scoring="r2", cv=3, n_jobs=n_jobs).mean()
            # score = -cross_val_score(
            #     model,
            #     X_train,
            #     y_train,
            #     cv=3,
            #     scoring="neg_root_mean_squared_error",
            #     n_jobs=n_jobs,
            # ).mean()
            return score

        catboost_bo = BayesianOptimization(
            catboost_evaluate,
            {
                "depth": (2, 8),
                "learning_rate": (0.01, 0.5),
                "l2_leaf_reg": (1, 9),
                "rsm": (0.5, 1),
                "subsample": (0.5, 1),
                "iterations": (10, 2000),
            },
            verbose=0,
            random_state=seed,
        )
        catboost_bo.maximize(init_points=10, n_iter=n_iter, acq="ei", xi=0.0)

        model = CatBoostRegressor(
            verbose=verbose,
            random_state=seed,
            depth=int(catboost_bo.max["params"]["depth"]),
            learning_rate=catboost_bo.max["params"]["learning_rate"],
            l2_leaf_reg=catboost_bo.max["params"]["l2_leaf_reg"],
            rsm=catboost_bo.max["params"]["rsm"],
            subsample=catboost_bo.max["params"]["subsample"],
            # iterations=catboost_bo.max["params"]["iterations"]
        )
    if custom_gradient:
        cat_train_data = Pool(data=X_train, label=y_train, weight=weights)
        model.fit(cat_train_data)
    else:
        if weights is not None:
            cat_train_data = Pool(data=X_train, label=y_train, weight=weights)
        else:
            cat_train_data = Pool(data=X_train, label=y_train)
        model.fit(cat_train_data)
    return model


class RmseObjectiveL1(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        der1 = weights * np.sign(targets - approxes)
        der2 = np.ones(len(targets)) * -1

        # companies_ids = pd.Series(companies_lst, name="FinalEikonID").to_frame()
        # for corpo_id in companies_ids.FinalEikonID:
        #     lst_idx = companies_ids[companies_ids.FinalEikonID == corpo_id].index
        #     corpo_der_sum = sum([der1[i] for i in lst_idx])  # use L1 norm, simplest to implement ?
        #     # corpo_der_sum = np.sqrt(sum([der1[i] ** 2 for i in range(lst_idx)]))  # use L2 norm, better properties ?
        #     # der1[i] = der1[i] / corpo_der_sum
        #     weights[lst_idx] = weights[lst_idx] / corpo_der_sum  # normalisation

        # for index in range(len(targets)):
        #     if weights is not None:
        #         der1[index] *= weights
        #         der2[index] *= weights

        result = np.array([der1, der2]).reshape(-1, 2)
        return result


class RmseObjectiveL2(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        der1 = weights * np.array([np.sign(targets[i] - approxes[i] for i in range(len(targets)))])
        der2 = np.ones(len(targets)) * -1

        # companies_ids = pd.Series(companies_lst, name="FinalEikonID").to_frame()
        # for corpo_id in companies_ids.FinalEikonID:
        #     lst_idx = companies_ids[companies_ids.FinalEikonID == corpo_id].index
        #     # corpo_der_sum = sum([der1[i] for i in lst_idx])  # use L1 norm, simplest to implement ?
        #     corpo_der_sum = np.sqrt(sum([der1[i] ** 2 for i in range(lst_idx)]))  # use L2 norm, better properties ?
        #     # der1[i] = der1[i] / corpo_der_sum
        #     weights[lst_idx] = weights[lst_idx] / corpo_der_sum  # normalisation

        # for index in range(len(targets)):
        #     if weights is not None:
        #         der1[index] *= weights
        #         der2[index] *= weights

        result = np.array([der1, der2]).reshape(-1, 2)
        return result


class RmseMetric(object):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        if weight:
            weight = weight
        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((approx[i] - target[i]) ** 2)

        return error_sum, weight_sum


def rmse_metric_lgbm(approxes, target, weight):
    is_higher_better = False

    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        if weight:
            weight = weight.weight_final.tolist()
        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((approx[i] - target[i]) ** 2)
        return error_sum, weight_sum

    return "custom_rmse_metric_lgbm", get_final_error(evaluate(approxes, target, weight)), is_higher_better
