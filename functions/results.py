import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
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
from functions.preprocessing import target_preprocessing, outliers_preprocess


def summary_detailed(X_test, y_test, y_pred, df_test, target, open_data):
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
    summary_region_sector = pd.DataFrame()
    deciles = pd.qcut(df_test["Revenue"], 10, labels=False)
    df_test_copy = df_test.copy()
    df_test_copy["Revenuebuckets"] = deciles
    # bucket_summary = df_test_copy.groupby("Revenuebuckets")["Revenue"].agg(["min", "max", "count", "mean"])
    # bucket_summary.to_csv("bucket_summary.csv")

    X_test_copy = X_test.copy()
    X_test_copy["y_pred"] = y_pred
    X_test_copy["y_test"] = y_test
    X_test_copy.loc[:, "SubSector"] = df_test_copy.GICSName
    X_test_copy.loc[:, "Revenuebuckets"] = df_test_copy.Revenuebuckets
    X_test_copy.loc[:, "Region"] = df_test_copy.Region

    if not open_data:
        X_test_copy.loc[:, "ENEConsume_log"] = df_test_copy.ENEConsume_log.isna()
        X_test_copy.loc[:, "ENEProduce_log"] = df_test_copy.ENEProduce_log.isna()
        X_test_copy.loc[:, "SectorName"] = df_test_copy.GICSSector
        categories = ["Region", "Revenuebuckets", "SubSector", "SectorName", "ENEConsume_log", "ENEProduce_log"]
    else:
        categories = ["Region", "Revenuebuckets", "SubSector"]

    for category in categories:
        unique = list(X_test_copy[category].unique())
        for value in unique:
            mask = X_test_copy[category] == value
            rmse = np.sqrt(mean_squared_error(X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"]))
            mse = mean_squared_error(X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"])
            mae = mean_absolute_error(X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"])
            mape = mean_absolute_percentage_error(X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"])
            r2 = r2_score(X_test_copy[mask]["y_test"], X_test_copy[mask]["y_pred"])
            std = np.std(X_test_copy[mask]["y_test"] - X_test_copy[mask]["y_pred"])

            summary = pd.DataFrame(
                {
                    "category": category,
                    "scope": target,
                    # "model_name": model_name,
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

            summary_region_sector = pd.concat([summary_region_sector, summary], ignore_index=True)

    return summary_region_sector


def plot(model, X, y_test, y_pred, plot_path, target):
    """
    This function generates three plots for evaluating the machine learning model's performance :

    A SHAP values plot showing the impact of each feature on the model's predictions.
    A scatter plot comparing the actual values against the predicted values.
    A residual plot showing the distribution of the difference between actual and predicted values.
    Parameters:

    model: The trained machine learning model to be evaluated.
    X: The feature matrix for the test dataset.
    y_test: The ground truth labels for the test dataset.
    y_pred: The predicted labels for the test dataset.
    plot_path: The directory where the plots will be saved.
    target: The name of the target variable.
    model_name: The name of the machine learning model.
    Returns:
    plots saved in plot_path
    """

    def plot_shap_values(model, X):
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X)
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X,
            feature_names=list(X.columns),
        )
        shap.plots.beeswarm(explanation, show=False, color_bar=False)
        plt.colorbar()
        plt.savefig(plot_path + f"shap_{target}.png")
        plt.show()

    def plot_y_test_y_pred(y_test, y_pred):
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.savefig(plot_path + f"y_test_y_pred_{target}.png")
        plt.show()

    def plot_residuals(y_test, y_pred):
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color="r", linestyle="-")
        plt.title("Residual Plot")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.savefig(plot_path + f"residus_{target}.png")
        plt.show()

    plot_shap_values(model, X)
    plot_y_test_y_pred(y_test, y_pred)
    plot_residuals(y_test, y_pred)


def scopes_report(
    dataset,
    target,
    best_model,
    estimated_scopes,
    path_intermediary,
    open_data,
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
        open_data=open_data,
    )
    final_dataset = set_columns(final_dataset, features)

    final_dataset_train = target_preprocessing(final_dataset, target)
    final_model = best_model.fit(final_dataset_train[features], final_dataset_train[target])
    final_y_pred = final_model.predict(final_dataset[features])

    with open(path_models + f"{target}_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    final_dataset_summary = final_dataset[lst]
    final_dataset_summary.loc[:, f"{target}_estimated"] = np.power(10, final_y_pred + 1)
    estimated_scopes.append(final_dataset_summary)
    return estimated_scopes, lst


def metrics(y_test, y_pred, Summary_Final, target, model_name, n_split=5):
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

    # std = np.std(y_test - y_pred)
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
    open_data,
    path_models,
):
    y_pred_best = best_model.predict(X_test)
    plot(best_model, X_train, y_test, y_pred_best, path_plot, target)
    metrics_scope = summary_detailed(X_test, y_test, y_pred_best, df_test, target, open_data)
    summary_metrics_detailed = pd.concat([summary_metrics_detailed, metrics_scope], ignore_index=True)

    estimated_scopes, lst = scopes_report(
        dataset,
        target,
        best_model,
        estimated_scopes,
        path_intermediary,
        open_data=open_data,
        path_models=path_models,
    )
    return summary_metrics_detailed, estimated_scopes, lst


def results(estimated_scopes, path_results, summary_metrics_detailed, Summary_Final, lst):
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
    summary_metrics_detailed.to_csv(path_results + "Summary_metrics_detail.csv", index=False)
    dict_recap = {}
    for key in ["Target", "model", "mae", "mse", "r2", "rmse", "mape", "std"]:
        dict_recap[key] = [Summary_Final[i][key] for i in range(len(Summary_Final))]
    df_Summary_Final = pd.DataFrame.from_dict(dict_recap)
    df_Summary_Final.to_csv(path_results + "Summary_metrics.csv", index=False)


def results_mlflow(estimated_scopes, path_results, summary_metrics, Summary_Final, lst, name_experiment):
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
    merged_df.to_csv(path_results + f"{name_experiment}_Estimated_scopes.csv", index=False)
    summary_metrics.to_csv(path_results + f"{name_experiment}_Summary_metrics_detail.csv", index=False)
    Summary_Final.to_csv(path_results + f"{name_experiment}_Summary_metrics.csv", index=False)
