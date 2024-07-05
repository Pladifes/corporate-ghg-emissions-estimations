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


def summary_detailed(
    X_test, y_test, y_pred, df_test, target, restricted_features, path_plot
):
    """
    Generates a detailed summary of the model performance by various categories such as sector, region, country, etc., for a given target variable.

    The function calculates and reports performance metrics like Root Mean Squared Error (RMSE), Mean Squared Error (MSE),
    Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), R-squared (R2), and Standard Deviation for each category.
    It also generates and saves plots for the detailed analysis of each category's performance.

    Note:
    - The categories considered for analysis may vary based on the presence of restricted features in the model.
    - Plots are saved in the specified 'path_plot' directory.

    Parameters:
    - X_test (pandas DataFrame): A DataFrame containing the test set features.
    - y_test (numpy array): An array with the actual values of the target variable for the test set.
    - y_pred (numpy array): An array with the predicted values for the target variable.
    - df_test (pandas DataFrame): A DataFrame containing the test set features and additional information
        about the sectors, regions, and other categories of the observations.
    - target (str): The name of the target variable column in the DataFrames.
    - restricted_features (list): A list of restricted features to be considered in the analysis.
    - path_plot (str): The path where plots and visualizations will be saved.

    Returns:
    - pandas DataFrame: A summary of the model performance by various categories. The DataFrame includes
        columns such as 'category', 'category_name', 'RMSE', 'MSE', 'MAE', 'MAPE', 'R2', and 'StandardDeviation'.
    """

    n_split = 10
    summary_region_sector = pd.DataFrame()
    deciles = pd.qcut(df_test["revenue"], 10, labels=False)
    df_test_copy = df_test.copy()
    df_test_copy["revenue_buckets"] = deciles

    X_test_copy = X_test.copy()
    X_test_copy["y_pred"] = y_pred
    X_test_copy["y_test"] = y_test
    X_test_copy["sub_sector"] = df_test_copy.gics_name.values
    X_test_copy["revenue_bucket"] = df_test_copy.revenue_buckets.values
    X_test_copy["region"] = df_test_copy.region.values
    X_test_copy["country"] = df_test_copy.country_hq.values
    X_test_copy["year"] = df_test_copy.fiscal_year.values

    if not restricted_features:
        df_test_copy["sectors"] = (
            df_test_copy["gics_sector"].astype(float).apply(gics_to_name)
        )
        X_test_copy["energy_consumed"] = df_test_copy.energy_consumed.isna().values
        X_test_copy["energy_produced"] = df_test_copy.energy_produced.isna().values
        X_test_copy["industry"] = df_test_copy.sectors.values
        categories = [
            "revenue_bucket",
            "region",
            "country",
            "industry",
            "sub_sector",
            "year",
            "energy_consumed",
            "energy_produced",
        ]
    else:
        categories = ["revenue_bucket", "region", "country", "sub_sector", "year"]

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

        if category in ["energy_consumed", "energy_produced"]:
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
    path_rawdata,
    customized_model,
):
    """
    This function generates a report of estimated scopes based on a provided dataset, target variable, best model, and additional parameters.

    Parameters:
    - dataset (pandas DataFrame): The dataset containing the relevant data for scope estimation.
    - target (str): The column name from the dataset to be used as the target variable in the model.
    - best_model (sklearn model object): The best-fit model object for the given dataset and features.
    - estimated_scopes (list): A list containing the estimated scopes for each dataset.
    - path_intermediary (str): The path to the intermediary files used in data processing.
    - restricted_features (list): A list of column names to be excluded as restricted features.
    - path_models (str): The path to store the trained model.

    Returns:
    - estimated_scopes (list): A list of estimated scopes, updated with the new dataset summary.
    - lst (list): A list of column names to be included in the final dataset summary.
    """
    features = pd.read_csv(path_intermediary + f"features_{target}.csv")
    features = features["features"].to_list()
    lst = [
        "company_id",
        "company_name",
        "ticker",
        # "lei",
        "fiscal_year",
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
        final_dataset_train[features], final_dataset_train[f"{target}_log"]
    )
    # dataset_predict = pd.read_parquet(path_rawdata + "predict_dataset.parquet")
    final_y_pred = final_model.predict(final_dataset[features])
    

    if customized_model : 
        df_uncertainty = uncertainty_est(final_dataset, final_y_pred, final_dataset[f"{target}_log"], confidence_multiplier=1.64)
        cols = ["y_lower", "y_upper", "y_mean"]
    
        for col in cols : 
            df_uncertainty[col] = np.power(10, df_uncertainty[col] + 1)
        
        columns_to_keep = ["company_id","company_name","ticker","fiscal_year","y_lower", "y_upper", "y_mean"]
        df_uncertainty = df_uncertainty[columns_to_keep]
        df_uncertainty.columns= ["company_id","company_name","ticker","fiscal_year",f"{target}_estimated_lower_bound",f"{target}_estimated_upper_bound",f"{target}_estimated"]
        final_y_pred =final_y_pred [:,0]

        with open(path_models + f"{target}_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        final_dataset_summary = final_dataset[lst]
        final_dataset_summary.loc[:, f"{target}_estimated"] = np.power(10, final_y_pred + 1)
        final_dataset_uncert = pd.merge(final_dataset_summary,df_uncertainty, on = lst + [f"{target}_estimated"], how = "left" )
        final_dataset_uncert= final_dataset_uncert.sort_values(by=["company_name","fiscal_year"], ascending=False)

        estimated_scopes.append(final_dataset_uncert)
    else : 
        with open(path_models + f"{target}_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        final_dataset_summary = final_dataset[lst]
        # final_dataset_summary = final_dataset[lst]
        final_dataset_summary.loc[:, f"{target}_estimated"] = np.power(10, final_y_pred + 1)
        estimated_scopes.append(final_dataset_summary)
    return estimated_scopes, lst



def metrics(y_test, y_pred, summary_final, target, model_name, n_split=10):
    """
    This function computes several evaluation metrics for a machine learning model's performance on a given dataset:

    R-squared (R2) score, which measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
    Mean absolute error (MAE), which measures the average absolute difference between the predicted and actual values.
    Mean squared error (MSE), which measures the average squared difference between the predicted and actual values.
    Root mean squared error (RMSE), which is the square root of the MSE.
    Mean absolute percentage error (MAPE), which measures the percentage difference between the predicted and actual values.
    Standard deviation (STD), which measures the spread of the difference between the predicted and actual values.

    Parameters:
    - y_test: The ground truth labels for the test dataset.
    - y_pred: The predicted labels for the test dataset.
    - summary_final: A list to store the summary of evaluation metrics for all models and targets.
    - target: The name of the target variable.
    - model_name: The name of the machine learning model.
    - n_split: The number of splits to calculate RMSE (root mean squared error) across different segments of the dataset. Defaults to 10.

    Returns:
    - summary_global: A pandas dataframe that contains the summary of evaluation metrics for all models and targets.
    - rmse: The root mean squared error (RMSE) value.
    - std: The standard deviation of RMSE values.

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
    summary_final.append(
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
    summary_global = pd.DataFrame(summary_final)
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
    path_rawdata,
    customized_model,
):
    """
    Analyze the performance of the best machine learning model and generate a detailed report.

    Parameters:
    - best_model (object): The trained machine learning model that performed the best.
    - X_test (DataFrame): The feature matrix of the test dataset.
    - X_train (DataFrame): The feature matrix of the training dataset.
    - y_test (Series): The true target values of the test dataset.
    - df_test (DataFrame): The test dataset.
    - target (str): The name of the target variable.
    - path_plot (str): The path to save generated plots and visualizations.
    - dataset (str): The name or description of the dataset.
    - path_intermediary (str): The path for storing intermediate analysis results.
    - summary_metrics_detailed (DataFrame): A DataFrame containing summary metrics for multiple models.
    - estimated_scopes (DataFrame): A DataFrame containing estimated scopes for models.
    - restricted_features (list): List of restricted features used in modeling.
    - path_models (str): The path to save model files.

    Returns:
    - summary_metrics_detailed (DataFrame): Updated summary metrics including the analysis of the best model.
    - estimated_scopes (DataFrame): Updated estimated scopes after analyzing the best model.
        lst (list): List of additional analysis results or information.


    """
    
    y_pred_best = best_model.predict(X_test)

    if customized_model : 
        y_pred_best =y_pred_best[:,0]

    plot(best_model, X_train, y_test, y_pred_best, path_plot, target, customized_model)

    metrics_scope = summary_detailed(
        X_test, y_test, y_pred_best, df_test, target, restricted_features, path_plot
    )

    summary_metrics_detailed = pd.concat(
        [summary_metrics_detailed, metrics_scope], ignore_index=True
    )
    # if not restricted_features:
    estimated_scopes, lst = scopes_report(
        dataset,
        target,
        best_model,
        estimated_scopes,
        path_intermediary,
        restricted_features=restricted_features,
        path_models=path_models,
        path_rawdata=path_rawdata,
        customized_model=True,
    )
    return summary_metrics_detailed, estimated_scopes, lst


def results(
    estimated_scopes, path_results, summary_metrics_detailed, summary_final, lst
):
    """
    Save the estimated scopes, summary metrics, and a summary report as files in the specified path.

    Parameters:
    - estimated_scopes (list): List of DataFrames containing the estimated scopes.
    - path_results (str): Path where the output files will be saved.
    - summary_metrics (DataFrame): DataFrame containing detailed summary metrics.
    - summary_final (DataFrame): DataFrame containing summary metrics.
    - lst (list): List of column names to merge DataFrames on.

    Returns:
    - None: This function doesn't return any values, it only saves files to the specified path.
    """
    nb_targets = len(estimated_scopes)
    merged_estimated_scopes = estimated_scopes[0]
    for k in range(1, nb_targets):
        merged_estimated_scopes = pd.merge(
            merged_estimated_scopes, estimated_scopes[k], on=lst, how="outer"
        )
    merged_estimated_scopes = merged_estimated_scopes.sort_values(
        by=["company_id"]
    )
    merged_estimated_scopes = merged_estimated_scopes.reset_index(drop=True)
    profile = ProfileReport(merged_estimated_scopes, minimal=True)
    profile.to_file(path_results + "scopes_summary.html")
    merged_estimated_scopes.to_csv(path_results + "estimated_scopes.csv", index=False)
    summary_metrics_detailed.to_csv(
        path_results + "summary_metrics_detail.csv", index=False
    )
    dict_recap = {}
    for key in ["Target", "model", "mae", "mse", "r2", "rmse", "mape", "std"]:
        dict_recap[key] = [summary_final[i][key] for i in range(len(summary_final))]
    df_summary_final = pd.DataFrame.from_dict(dict_recap)
    df_summary_final.to_csv(path_results + "summary_metrics.csv", index=False)


def gics_to_name(gics_sector):
    """
    Converts a GICS (Global Industry Classification Standard) sector code into the corresponding sector name.

    Parameters:
    - gics_sector (float): The GICS sector code to be converted.

    Returns:
    - str: The name of the GICS sector corresponding to the input code. If the input code does not match
             any known GICS sector, the input code itself is returned as a string.
    """

    if gics_sector == 10.0:
        return "Energy"
    elif gics_sector == 15.0:
        return "Materials"
    elif gics_sector == 20.0:
        return "Industrials"
    elif gics_sector == 25.0:
        return "Cons. Discretionary"
    elif gics_sector == 30.0:
        return "Cons. Staples"
    elif gics_sector == 35.0:
        return "Health Care"
    elif gics_sector == 40.0:
        return "Financials"
    elif gics_sector == 45.0:
        return "IT"
    elif gics_sector == 50.0:
        return "Telecommunication"
    elif gics_sector == 55.0:
        return "Utilities"
    elif gics_sector == 60.0:
        return "Real Estate"
    else:
        return gics_sector


def is_between(y_true, y_lower, y_upper):
    return (y_true >= y_lower) and (y_true <= y_upper)

def uncertainty_est(initial_dataset, y_pred, y_test, confidence_multiplier):


    mean_y_pred, var_y_pred = y_pred[:,0], y_pred[:,1]
    df_test =  pd.DataFrame({
        "company_id": initial_dataset["company_id"],
        "company_name": initial_dataset["company_name"],
        "ticker": initial_dataset["ticker"],
        "fiscal_year": initial_dataset["fiscal_year"],
        "y_true": y_test,
        "y_mean":mean_y_pred,
        "y_variance": var_y_pred})
    
    df_test["y_lower"] = (df_test["y_mean"] - confidence_multiplier * np.sqrt(df_test["y_variance"]))

    df_test["y_upper"] = (df_test["y_mean"] + confidence_multiplier * np.sqrt(df_test["y_variance"]))

    df_test["is_between"] = df_test.apply(lambda x: is_between(x["y_true"], x["y_lower"], x["y_upper"]), axis=1)

    df_test["difference"] = df_test["y_upper"] - df_test["y_lower"]
    return df_test

def uncertainty_analysis(df_train, df_test, target, difference_mean_train_dict,couvrage_train_dict, couvrage_test_dict, difference_mean_test_dict):

    couvrage_train = df_train["is_between"].sum()/len(df_train)
    couvrage_train_dict[target] = couvrage_train
    couvrage_test = df_test["is_between"].sum()/len(df_test)
    couvrage_test_dict[target] = couvrage_test
    difference_mean_train_dict[target] = df_train["difference"].mean()
    difference_mean_test_dict[target] = df_test["difference"].mean()

    df_couvrage = pd.DataFrame({
        "Covrage train": couvrage_train_dict.values(),
        "Covrage test": couvrage_test_dict.values(),
        "Difference mean train": difference_mean_train_dict.values(),
        "Difference mean test": difference_mean_test_dict.values()
        })

    return df_couvrage

