import numpy as np

from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from bayes_opt import BayesianOptimization


def xgboost_model(
    X_train,
    y_train,
    cross_val=False,
    verbose=0,
    n_jobs=-1,
    n_iter=10,
    seed=None,
    weights=None,
    custom_gradient=False,
):
    """
    Create an XGBoost regression model.

    Parameters:
    - X_train (array-like or pd.DataFrame): The training data.
    - y_train (array-like or pd.Series): The target variable for training.
    - cross_val (bool): If True, perform Bayesian hyperparameter optimization using cross-validation.
    - verbose (int): Controls the verbosity of the XGBoost model (0 for silent).
    - n_jobs (int): Number of CPU cores to use for parallel execution during cross-validation.
    - n_iter (int): Number of iterations for Bayesian optimization if cross-validation is enabled.
    - seed (int): Random seed for reproducibility.
    - weights (array-like): Sample weights for training data (optional).
    - custom_gradient (bool): If True, enable custom gradient (not supported by XGBoost).

    Returns:
    - model (XGBRegressor): The trained XGBoost regression model.
    """
    model = XGBRegressor(random_state=seed, verbosity=0)

    if cross_val:

        def xgb_evaluate(
            n_estimators, learning_rate, max_depth, gamma, subsample, colsample_bytree
        ):
            params = {
                "n_estimators": int(n_estimators),
                "learning_rate": learning_rate,
                "max_depth": int(max_depth),
                "gamma": gamma,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }

            model = XGBRegressor(verbose=verbose, random_state=seed, **params)
            score = cross_val_score(
                model, X_train, y_train, scoring="r2", cv=3, n_jobs=n_jobs
            ).mean()
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
    X_train,
    y_train,
    cross_val=False,
    n_jobs=-1,
    verbose=-1,
    n_iter=10,
    seed=None,
    weights=None,
    custom_gradient=False,
):
    """
    Create a LightGBM (LGBM) regression model.

    Parameters:
    - X_train (array-like or pd.DataFrame): The training data.
    - y_train (array-like or pd.Series): The target variable for training.
    - cross_val (bool): If True, perform Bayesian hyperparameter optimization using cross-validation.
    - n_jobs (int): Number of CPU cores to use for parallel execution during cross-validation.
    - verbose (int): Controls the verbosity of the LGBM model (higher values for more information).
    - n_iter (int): Number of iterations for Bayesian optimization if cross-validation is enabled.
    - seed (int): Random seed for reproducibility.
    - weights (array-like): Sample weights for training data (optional).
    - custom_gradient (bool): If True, enable custom gradient (not supported by LightGBM).

    Returns:
    - model (LGBMRegressor): The trained LightGBM regression model.
    """
    verbose -= 1
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
            score = cross_val_score(
                model, X_train, y_train, scoring="r2", cv=3, n_jobs=n_jobs
            ).mean()
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

    model.fit(X_train, y_train)

    return model


def catboost_model(
    X_train,
    y_train,
    cross_val=False,
    n_jobs=-1,
    verbose=0,
    n_iter=10,
    seed=None,
    weights=None,
    custom_gradient=False,
):
    """
    Create a CatBoost regression model.

    Parameters:
    - X_train (array-like or pd.DataFrame): The training data.
    - y_train (array-like or pd.Series): The target variable for training.
    - cross_val (bool): If True, perform Bayesian hyperparameter optimization using cross-validation.
    - n_jobs (int): Number of CPU cores to use for parallel execution during cross-validation.
    - verbose (int): Controls the verbosity of the CatBoost model (higher values for more information).
    - n_iter (int): Number of iterations for Bayesian optimization if cross-validation is enabled.
    - seed (int): Random seed for reproducibility.
    - weights (array-like): Sample weights for training data (optional).
    - custom_gradient (bool): If True, enable custom gradient (not supported by CatBoost).

    Returns:
    - model (CatBoostRegressor): The trained CatBoost regression model.
    """

    model = CatBoostRegressor(verbose=verbose, random_state=seed)

    if cross_val:

        def catboost_evaluate(
            depth, learning_rate, l2_leaf_reg, rsm, subsample, iterations
        ):
            params = {
                "depth": int(depth),
                "learning_rate": learning_rate,
                "l2_leaf_reg": l2_leaf_reg,
                "rsm": rsm,
                "subsample": subsample,
                "iterations": int(iterations),
            }
            model = CatBoostRegressor(verbose=verbose, random_state=seed, **params)
            score = cross_val_score(
                model, X_train, y_train, scoring="r2", cv=10, n_jobs=n_jobs
            ).mean()
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

    cat_train_data = Pool(data=X_train, label=y_train)
    model.fit(cat_train_data)

    return model
