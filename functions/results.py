import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from pandas_profiling import ProfileReport

from functions.preprocessing import encoding
from functions.preprocessing import set_columns
from functions.preprocessing import target_preprocessing
from functions.plot_functions import plot, plot_detailed


def GICS_to_name(GICSSector):
    """
    Convert GICS (Global Industry Classification Standard) sector code to sector name.

    Parameters:
    - GICSSector (float): The GICS sector code to be converted to a sector name.

    Returns:
    - str: The name of the sector corresponding to the given GICS sector code.

    This function takes a GICS sector code as input and returns the corresponding sector name.
    The GICS system is used to classify companies into different industry sectors.
    If the input GICSSector matches one of the predefined sector codes (e.g., 10.0 for Energy),
    the function returns the corresponding sector name. If the input does not match any predefined code,
    it returns the input GICSSector as a string.

    """
    if GICSSector == 10.0:
        return "Energy"
    elif GICSSector == 15.0:
        return "Materials"
    elif GICSSector == 20.0:
        return "Industrials"
    elif GICSSector == 25.0:
        return "Cons. Discretionary"
    elif GICSSector == 30.0:
        return "Cons. Staples"
    elif GICSSector == 35.0:
        return "Health Care"
    elif GICSSector == 40.0:
        return "Financials"
    elif GICSSector == 45.0:
        return "IT"
    elif GICSSector == 50.0:
        return "Telecommunication"
    elif GICSSector == 55.0:
        return "Utilities"
    elif GICSSector == 60.0:
        return "Real Estate"
    else:
        return GICSSector


def summary_detailed(
    X_test, y_test, y_pred, df_test, target, restricted_features, path_plot
):
    """
    Generates a summary of the model performance by sector and region for a given target variable.

    Args:
        X_test: A pandas DataFrame containing the test set features.
        y_pred: A numpy array with the predicted values for the target variable.
        y_test: A numpy array with the actual values of the target variable for the test set.
        df_test: A pandas DataFrame containing the test set features and additional information
            about the sectors and regions of the observations.
        target: A string with the name of the target variable column in the DataFrames.

    Returns:
        A pandas DataFrame with the summary statistics of the model performance by sector and region.
        The DataFrame contains the following columns:
        - 'category',
        - 'category_name',
        - 'RMSE',
        - 'MSE',
        - 'MAE',
        - 'MAPE',
        - "R2",
        - scope"
    """
    n_split = 10
    summary_region_sector = pd.DataFrame()
    deciles = pd.qcut(df_test["Revenue"], 10, labels=False)
    df_test_copy = df_test.copy()
    df_test_copy["Revenuebuckets"] = deciles
    df_test_copy["Sectors"] = (
        df_test_copy["GICSSector"].astype(float).apply(GICS_to_name)
    )

    X_test_copy = X_test.copy()
    X_test_copy["y_pred"] = y_pred
    X_test_copy["y_test"] = y_test
    X_test_copy["SubSector"] = df_test_copy.GICSName.values
    X_test_copy["Revenuebucket"] = df_test_copy.Revenuebuckets.values
    X_test_copy["Region"] = df_test_copy.Region.values
    X_test_copy["Country"] = df_test_copy.CountryHQ.values
    X_test_copy["Year"] = df_test_copy.FiscalYear.values

    if not restricted_features:
        X_test_copy["ENEConsume"] = df_test_copy.ENEConsume_y.isna().values
        X_test_copy["ENEProduce"] = df_test_copy.ENEProduce_y.isna().values
        X_test_copy["Industry"] = df_test_copy.Sectors.values
        categories = [
            "Revenuebucket",
            "Region",
            "Country",
            "Industry",
            "SubSector",
            "Year",
            "ENEConsume",
            "ENEProduce",
        ]
    else:
        categories = ["Revenuebucket", "Region", "Country", "SubSector", "Year"]

    for category in categories:
        rmses_df = pd.DataFrame([], columns=["rmses", category])
        unique = list(X_test_copy[category].unique())
        for i, value in enumerate(unique):
            mask = X_test_copy[category] == value
            rmse = np.sqrt(
                mean_squared_error(
                    X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"]
                )
            )
            mse = mean_squared_error(
                X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"]
            )
            mae = mean_absolute_error(
                X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"]
            )
            mape = mean_absolute_percentage_error(
                X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"]
            )
            r2 = r2_score(X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"])
            rmses = []
            l_test = len(X_test_copy[mask]["y_test"])
            if len(X_test_copy[mask]) >= n_split:
                for k in range(n_split):
                    rmses.append(
                        mean_squared_error(
                            X_test_copy[mask]["y_test"][
                                int(k * l_test / n_split) : int(
                                    (k + 1) * l_test / n_split
                                )
                            ],
                            X_test_copy[mask]["y_pred"][
                                int(k * l_test / n_split) : int(
                                    (k + 1) * l_test / n_split
                                )
                            ],
                            squared=False,
                        )
                    )
                std = np.std(rmses)
                rmses_df_temp = pd.DataFrame(
                    [[rmse, value] for rmse in rmses], columns=["rmses", category]
                )
                rmses_df = pd.concat([rmses_df, rmses_df_temp])
            else:
                std = np.nan

            summary = pd.DataFrame(
                {
                    "category": category,
                    "scope": target,
                    "category_name": value,
                    "RMSE": rmse,
                    "MSE": mse,
                    "MAE": mae,
                    "MAPE": mape,
                    "R2": r2,
                    "StandardDeviation": std,
                },
                index=[0],
            )

            summary_region_sector = pd.concat(
                [summary_region_sector, summary], ignore_index=True
            )

        if category in ["ENEConsume", "ENEProduce"]:
            rmses_df = rmses_df.replace({True: "No values", False: "With values"})
        sorted_categs = (
            rmses_df.groupby(category)
            .median()
            .sort_values(by="rmses", ascending=False)
            .index
        )
        categ_type = pd.CategoricalDtype(categories=sorted_categs, ordered=True)
        rmses_df[category] = pd.Series(rmses_df[category], dtype=categ_type)
        rmses_df = rmses_df.sort_values(by=[category])
        plot_detailed(rmses_df, target, path_plot, category)

    return summary_region_sector


def scopes_report(
    dataset,
    target,
    best_model,
    estimated_scopes,
    path_intermediary,
    restricted_features,
    path_models,
):
    """
    This function generates a report of estimated scopes based on a provided dataset, features, target, and best model.

    Parameters:

    dataset (pandas DataFrame): The dataset containing the relevant data for scope estimation
    features (list): A list of column names from the dataset to be used as the features in the model
    target (str): The column name from the dataset to be used as the target variable in the model
    best_model (sklearn model object): The best-fit model object for the given dataset and features
    estimated_scopes (list): A list containing the estimated scopes for each dataset
    lst (list): A list of column names to be included in the final dataset summary
    Returns:

    estimated_scopes (list): A list of estimated scopes, updated with the new dataset summary
    """
    features = pd.read_csv(path_intermediary + "features.csv")
    features = features["features"].to_list()
    lst = [
        "FinalEikonID",
        "FiscalYear",
    ]

    final_dataset = encoding(
        dataset,
        path_intermediary,
        train=True,
        restricted_features=restricted_features,
    )
    final_dataset = set_columns(final_dataset, features)

    final_dataset_train = target_preprocessing(final_dataset, target)
    final_model = best_model.fit(
        final_dataset_train[features], final_dataset_train[target]
    )
    final_y_pred = final_model.predict(final_dataset[features])

    with open(path_models + f"{target}_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    final_dataset_summary = final_dataset[lst]
    final_dataset_summary.loc[:, f"{target}_estimated"] = np.power(10, final_y_pred + 1)
    estimated_scopes.append(final_dataset_summary)
    return estimated_scopes, lst


def metrics(y_test, y_pred, Summary_Final, target, model_name, n_split=10):
    """
    This function computes several evaluation metrics for a machine learning model's performance on a given dataset:

    R-squared (R2) score, which measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
    Mean absolute error (MAE), which measures the average absolute difference between the predicted and actual values.
    Mean squared error (MSE), which measures the average squared difference between the predicted and actual values.
    Root mean squared error (RMSE), which is the square root of the MSE.
    Mean absolute percentage error (MAPE), which measures the percentage difference between the predicted and actual values.
    Standard deviation (STD), which measures the spread of the difference between the predicted and actual values.
    Parameters:

    y_test: The ground truth labels for the test dataset.
    y_pred: The predicted labels for the test dataset.
    Summary_Final: A list to store the summary of evaluation metrics for all models and targets.
    target: The name of the target variable.
    model_name: The name of the machine learning model.
    Returns:

    summary_global: A pandas dataframe that contains the summary of evaluation metrics for all models and targets.
    rmse: The root mean squared error (RMSE) value.

    """
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmses = []
    l_test = len(y_test)
    for k in range(n_split):
        rmses.append(
            mean_squared_error(
                y_test[int(k * l_test / n_split) : int((k + 1) * l_test / n_split)],
                y_pred[int(k * l_test / n_split) : int((k + 1) * l_test / n_split)],
                squared=False,
            )
        )
    std = np.std(rmses)
    Summary_Final.append(
        {
            "Target": target,
            "model": model_name,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "rmse": rmse,
            "mape": mape,
            "std": std,
        }
    )
    summary_global = pd.DataFrame(Summary_Final)
    return summary_global, rmse, std


def best_model_analysis(
    best_model,
    X_test,
    X_train,
    y_test,
    df_test,
    target,
    path_plot,
    dataset,
    path_intermediary,
    summary_metrics_detailed,
    estimated_scopes,
    restricted_features,
    path_models,
):
    """
    Analyze the performance of the best machine learning model and generate a detailed report.

    Parameters:
        best_model (object): The trained machine learning model that performed the best.
        X_test (DataFrame): The feature matrix of the test dataset.
        X_train (DataFrame): The feature matrix of the training dataset.
        y_test (Series): The true target values of the test dataset.
        df_test (DataFrame): The test dataset.
        target (str): The name of the target variable.
        path_plot (str): The path to save generated plots and visualizations.
        dataset (str): The name or description of the dataset.
        path_intermediary (str): The path for storing intermediate analysis results.
        summary_metrics_detailed (DataFrame): A DataFrame containing summary metrics for multiple models.
        estimated_scopes (DataFrame): A DataFrame containing estimated scopes for models.
        restricted_features (list): List of restricted features used in modeling.
        path_models (str): The path to save model files.

    Returns:
        summary_metrics_detailed (DataFrame): Updated summary metrics including the analysis of the best model.
        estimated_scopes (DataFrame): Updated estimated scopes after analyzing the best model.
        lst (list): List of additional analysis results or information.

    This function evaluates the best machine learning model's performance on the test dataset,
    generates relevant plots, computes detailed metrics, and updates the summary and scope information.

    """
    y_pred_best = best_model.predict(X_test)
    plot(best_model, X_train, y_test, y_pred_best, path_plot, target)
    metrics_scope = summary_detailed(
        X_test, y_test, y_pred_best, df_test, target, restricted_features, path_plot
    )
    summary_metrics_detailed = pd.concat(
        [summary_metrics_detailed, metrics_scope], ignore_index=True
    )

    estimated_scopes, lst = scopes_report(
        dataset,
        target,
        best_model,
        estimated_scopes,
        path_intermediary,
        restricted_features=restricted_features,
        path_models=path_models,
    )
    return summary_metrics_detailed, estimated_scopes, lst


def results(
    estimated_scopes, path_results, summary_metrics_detailed, Summary_Final, lst
):
    """
    Save the estimated scopes, summary metrics, and a summary report as files in the specified path.

    Parameters:
    estimated_scopes (list): List of DataFrames containing the estimated scopes.
    path_results (str): Path where the output files will be saved.
    summary_metrics (DataFrame): DataFrame containing detailed summary metrics.
    Summary_Final (DataFrame): DataFrame containing summary metrics.
    lst (list): List of column names to merge DataFrames on.

    Returns:
    None: This function doesn't return any values, it only saves files to the specified path.
    """
    merged_df = pd.merge(estimated_scopes[0], estimated_scopes[1], on=lst, how="outer")
    merged_df = pd.merge(merged_df, estimated_scopes[2], on=lst, how="outer")
    merged_df = pd.merge(merged_df, estimated_scopes[3], on=lst, how="outer")
    merged_df = merged_df.sort_values(by=["FinalEikonID", "FiscalYear"])
    merged_df = merged_df.reset_index(drop=True)
    profile = ProfileReport(merged_df, minimal=True)
    profile.to_file(path_results + "Scopes_summary.html")
    merged_df.to_csv(path_results + "Estimated_scopes.csv", index=False)
    summary_metrics_detailed.to_csv(
        path_results + "Summary_metrics_detail.csv", index=False
    )
    dict_recap = {}
    for key in ["Target", "model", "mae", "mse", "r2", "rmse", "mape", "std"]:
        dict_recap[key] = [Summary_Final[i][key] for i in range(len(Summary_Final))]
    df_Summary_Final = pd.DataFrame.from_dict(dict_recap)
    df_Summary_Final.to_csv(path_results + "Summary_metrics.csv", index=False)


def results_mlflow(
    estimated_scopes, path_results, summary_metrics, Summary_Final, lst, name_experiment
):
    """
    Save the estimated scopes, summary metrics, and a summary report as files in the specified path.

    Parameters:
    estimated_scopes (list): List of DataFrames containing the estimated scopes.
    path_results (str): Path where the output files will be saved.
    summary_metrics (DataFrame): DataFrame containing detailed summary metrics.
    Summary_Final (DataFrame): DataFrame containing summary metrics.
    lst (list): List of column names to merge DataFrames on.

    Returns:
    None: This function doesn't return any values, it only saves files to the specified path.
    """
    merged_df = pd.merge(estimated_scopes[0], estimated_scopes[1], on=lst, how="outer")
    merged_df = pd.merge(merged_df, estimated_scopes[2], on=lst, how="outer")
    merged_df = pd.merge(merged_df, estimated_scopes[-1], on=lst, how="outer")
    merged_df.sort_values(by=["FinalEikonID", "FiscalYear"]).reset_index(drop=True)
    profile = ProfileReport(merged_df, minimal=True)
    profile.to_file(path_results + "Scopes_summary.html")
    merged_df.to_csv(
        path_results + f"{name_experiment}_Estimated_scopes.csv", index=False
    )
    summary_metrics.to_csv(
        path_results + f"{name_experiment}_Summary_metrics_detail.csv", index=False
    )
    Summary_Final.to_csv(
        path_results + f"{name_experiment}_Summary_metrics.csv", index=False
    )
