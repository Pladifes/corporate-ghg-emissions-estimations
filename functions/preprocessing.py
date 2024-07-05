import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import warnings
import json

warnings.simplefilter(action="ignore", category=FutureWarning)


def target_preprocessing(df, target):
    """
    A function to perform a target specific preprocessing on a pandas DataFrame.

    Parameters:
    - df (pandas.DataFrame): A pandas DataFrame containing the data.
    - target (str): The target variable to be used in the analysis.

    Returns:
    - pandas.DataFrame: A preprocessed DataFrame.
    """
    if target == "cf1":
        df = df[df["fiscal_year"] >= 2005]
        df = df.dropna(subset=["cf1_log"])
    elif target == "cf2":
        df = df[df["fiscal_year"] >= 2005]
        df = df.dropna(subset=["cf2_log"])
    elif target == "cf3":
        df = df[df["fiscal_year"] >= 2011]
        df = df.dropna(subset=["cf3_log"])
    elif (
        (target == "cf123")
        or (target == "cf1_cf123")
        or (target == "cf2_cf123")
        or (target == "cf3_cf123")
    ):
        df = df[df["fiscal_year"] >= 2011]
        df = df.dropna(subset=["cf123_log"])

    return df


def df_split(df, path_benchmark):
    """
    Split a pandas DataFrame into two parts based on company names.

    Parameters:
    - df (pandas DataFrame): The DataFrame to be split.
    - path_benchmark (str): The path to the benchmark file containing a list of company names.

    Returns:
    - tuple of pandas DataFrames: A tuple containing two DataFrames. The first DataFrame includes rows from the input DataFrame with company names not found in the benchmark list, and the second DataFrame includes rows with company names found in the benchmark list.

    This function splits the input DataFrame into two based on the presence of company names in a benchmark list. Rows with company names found in the benchmark list are included in the second DataFrame, while rows with company names not in the benchmark list are included in the first DataFrame.

    If the benchmark file is not found, the function randomly selects 20% of the data as a test set and ensures that test sectors are also present in the training set.
    """
    try:
        benchmark = pd.read_csv(path_benchmark + "lst_companies_test.csv")
        benchmark_lst = benchmark["company_name"].tolist()
        mask = df["company_name"].isin(benchmark_lst)
        df_test = df[mask]
        df_train = df[~mask]
    except FileNotFoundError:
        # Take 20% of data if no benchmark is available
        names_shuffle = df["company_name"].sample(frac=1)[: int(len(df) / 5)]
        mask = df["company_name"].isin(names_shuffle)
        df_test = df[mask]
        df_train = df[~mask]
        for sector in df_test.gics_name.unique():
            if sector not in df_train.gics_name.unique():
                df_test = df_test[df_test.gics_name != sector]
    return df_train, df_test


def set_columns(df, features):
    """
    Align the column names of two pandas DataFrames and replace special characters in column names with underscores.

    Parameters:
    - df_train: pandas DataFrame, containing training data.
    - df_test: pandas DataFrame, containing testing data.

    Returns:
    - df_train: pandas DataFrame, containing training data with aligned and cleaned column names.
    - df_test: pandas DataFrame, containing testing data with aligned and cleaned column names.
    """

    missing_cols = list(set(features) - set(df.columns))
    if missing_cols:
        df = pd.concat(
            [df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1
        )
    return df


def logtransform(df, ls, path_results, train):
    """
    Calculate the base-10 logarithm of selected columns in a DataFrame and return the updated DataFrame.

    This function calculates the logarithm of base 10 for the values in the selected columns.
    If 'train' is True, the minimum values of these columns are saved to a CSV file for later use.
    If 'train' is False, the function retrieves the minimum values from the CSV file and uses them for transformation.

    Note:
    - Logarithmic transformation is applied as log10(x) for each specified column in 'ls'.
    - A minimum value is subtracted from the column before the transformation to ensure positive results.
    - If a column has values less than the minimum (which can occur due to transformation with 'train' as False), the logarithm is set to zero.

    Parameters:
    - df (pandas DataFrame): The input DataFrame containing the data to be transformed.
    - ls (list of str): A list of column names to be logarithmically transformed.
    - path_results (str): The path to save the output CSV file containing minimum values for each transformed column.
    - train (bool, optional): A flag indicating if the function is being used for training (default is True).

    Returns:
    - df (pandas DataFrame): The DataFrame with specified columns logarithmically transformed.
    """
    res = df.copy()
    columns_min = []

    if train:
        for l in ls:
            if l in ["cf1", "cf3", "cf2", "cf123"]:
                res[l + "_log"] = np.log10(res[l] + 1)
            else:
                res[l + "_log"] = np.log10(res[l] - res[l].min() + 1)
            columns_min_dict = {
                "column": l,
                "min_value": res[l].min(),
                
            }
            columns_min.append(columns_min_dict)

        columns_min_df = pd.DataFrame(columns_min)
        columns_min_df.to_csv(path_results + f"columns_min.csv", index=False)

    else:
        columns_min = pd.read_csv(path_results + f"columns_min.csv")
        for l in ls:
            if l in ["cf1", "cf3", "cf2", "cf123"]:
                res[l + "_log"] = np.log10(res[l] + 1)
            else:
                min_l = columns_min.loc[columns_min["column"] == l, "min_value"].item()
                if min_l <= res[l].min():
                    res[l + "_log"] = np.log10(res[l] - min_l + 1)
                else:
                    res.loc[
                        res[res[l] < min_l].index, l + "_log"
                    ] = 0  # = log(1) = log(min_l - min_l  + 1)
                    res.loc[res[res[l] >= min_l].index, l + "_log"] = np.log10(
                        res.loc[res[res[l] >= min_l].index, l] - min_l + 1
                    )
    return res


def label_final_co2_law(row):
    """
    This function applies a label to a row in a pandas DataFrame based on the values in the "CO2Law", "CO2Status", and "CO2Coverage" columns.

    Parameters:
   -  row (pandas.Series): The row to be labeled.

    Returns:
    - str: The label for the row.

    """
    if row["co2_law"] == "Yes" and row["co2_status"] == "Implemented":
        return "has_implemented_CO2Law"
    else:
        return "has_not_implemented_CO2Law"


def processingrawdata(data_old, restricted_features, train):
    """
    Process raw data by applying various transformations and generating dummy variables.

    This function applies data processing and encoding to the input DataFrame, generating dummy variables for categorical features and ordinal encoding for certain variables. It also handles missing values and ensures that data types are appropriately formatted.

    If 'restricted_features' is False, the function generates dummy variables for 'gics_sector', 'gics_group', 'gics_ind', and 'gics_sub_ind' columns to represent categorical information.

    If 'train' is True, the function encodes the 'income_group' and 'final_co2_law' columns using ordinal encoding and creates new columns for the encoded values.

    Note:
    - The specific encoding and dummy variable generation may vary based on the presence of restricted features.
    
    Parameters:
    - data_old (pandas DataFrame): The raw data to be processed.
    - restricted_features (bool): A flag indicating whether restricted features are applied.
    - train (bool): A flag indicating whether the data is part of the training set.

    Returns:
    - data_new (pandas DataFrame): The processed data with transformed and encoded columns.
    """
    data_new = data_old.copy()
    data_new["final_co2_law"] = data_new.apply(
        lambda row: label_final_co2_law(row), axis=1
    )

    if not restricted_features or train :
        data_new["gics_group"] = data_new["gics_group"].astype(str)
        data_new["gics_sector"] = data_new["gics_sector"].astype(str)
        data_new["gics_ind"] = data_new["gics_ind"].astype(str)

    data_new["gics_sub_ind"] = data_new["gics_sub_ind"].astype(str)

    data_new["income_group"] = data_new["income_group"].fillna(
        data_new["income_group"].value_counts().index[0]
    )
    income_group_encoder = OrdinalEncoder(categories=[["H", "UM", "LM", "L"]])
    data_new["income_group_encoded"] = income_group_encoder.fit_transform(
        data_new[["income_group"]]
    )

    final_co2_law_encoder = OrdinalEncoder()
    data_new["final_co2_law_encoded"] = final_co2_law_encoder.fit_transform(
        data_new[["final_co2_law"]]
    )

    if not restricted_features:
        data_new = pd.concat(
            [data_new, pd.get_dummies(data_new["gics_sector"], prefix="gics_sector")],
            axis=1,
        )
        data_new = pd.concat(
            [data_new, pd.get_dummies(data_new["gics_group"], prefix="gics_group")],
            axis=1,
        )
        data_new = pd.concat(
            [data_new, pd.get_dummies(data_new["gics_ind"], prefix="gics_ind_")], axis=1
        )

    data_new = pd.concat(
        [data_new, pd.get_dummies(data_new["gics_sub_ind"], prefix="gics_sub_ind")],
        axis=1,
    )

    return data_new


def fillmeanindustry(data_old, columnlist, path_intermediary, train):
    """
    Fill in missing values in a pandas DataFrame by using the mean of the corresponding group values.

    This function fills missing values in the input DataFrame by utilizing the mean values of corresponding groups based on certain criteria.

    If 'train' is set to True, the function calculates the means for specified 'columnlist' within groups defined by 'gics_sub_ind' and 'revenue' percentiles. The calculated means are stored in a dictionary and saved to a JSON file in the 'path_intermediary' directory.

    If 'train' is False, the function retrieves the means from the saved JSON file and uses them to fill missing values in 'columnlist' based on the 'gics_sub_ind' and 'revenue' percentiles.

    Note:
    - The function handles missing values by grouping data based on 'gics_sub_ind' and 'revenue' percentiles and filling missing values with the mean of the corresponding group.
    - The specific columns in 'columnlist' may vary based on the application.
    - The 'path_intermediary' is used to store and retrieve intermediary files for data processing.

    
    Parameters:
    - data_old (pandas DataFrame): The input data to be processed.
    - columnlist (list of strings): A list of column names for which missing values should be filled.
    - path_intermediary (str): The path to intermediary files used in data processing.
    - train (bool): A boolean flag indicating whether the data is part of the training set.

    Returns:
    - data_new (pandas DataFrame): The processed data with missing values filled in.
    """
    data_new = data_old.copy()
    if train:
        nb_per_sub_ind = data_new.groupby(["gics_sub_ind"])[columnlist].count()
        dict_mean_to_impute = pd.DataFrame(columns=columnlist).to_dict()

        for column in columnlist:
            dict_mean_to_impute_col = {}

            for sub_ind in nb_per_sub_ind[column].index:
                index_to_fill = data_new[data_new.gics_sub_ind == sub_ind].index
                data_temp = data_new[data_new.gics_sub_ind == sub_ind][
                    [column, "revenue"]
                ]
                data_temp["revenue_Bkt"] = pd.qcut(
                    data_temp.revenue, 10, duplicates="drop"
                )

                if data_temp.groupby("revenue_Bkt").count()[column].min() < 10:
                    ind = sub_ind[:-4] + sub_ind[-2:]
                    data_temp = data_new[data_new.gics_ind == ind][[column, "revenue"]]
                    data_temp["revenue_Bkt"] = pd.qcut(
                        data_temp.revenue, 10, duplicates="drop"
                    )

                    if data_temp.groupby("revenue_Bkt").count()[column].min() < 10:
                        grp = ind[:-4] + ind[-2:]
                        data_temp = data_new[data_new.gics_group == grp][
                            [column, "revenue"]
                        ]
                        data_temp["revenue_Bkt"] = pd.qcut(
                            data_temp.revenue, 10, duplicates="drop"
                        )

                        if data_temp.groupby("revenue_Bkt").count()[column].min() < 10:
                            sect = grp[:-4] + grp[-2:]
                            data_temp = data_new[data_new.gics_sector == sect][
                                [column, "revenue"]
                            ]
                            data_temp["revenue_Bkt"] = pd.qcut(
                                data_temp.revenue, 10, duplicates="drop"
                            )
                            filled_sub_ind_values = data_new.loc[
                                index_to_fill, column
                            ].fillna(
                                data_temp.groupby("revenue_Bkt")[column].transform(
                                    "mean"
                                )
                            )
                            data_new.loc[index_to_fill, column] = filled_sub_ind_values

                            temp_means = data_temp.groupby("revenue_Bkt")[column].mean()
                            temp_means.index = temp_means.index.astype(str)
                            dict_mean_to_impute_col[sub_ind] = temp_means.to_dict()

                        else:
                            filled_sub_ind_values = data_new.loc[
                                index_to_fill, column
                            ].fillna(
                                data_temp.groupby("revenue_Bkt")[column].transform(
                                    "mean"
                                )
                            )
                            data_new.loc[index_to_fill, column] = filled_sub_ind_values

                            temp_means = data_temp.groupby("revenue_Bkt")[column].mean()
                            temp_means.index = temp_means.index.astype(str)
                            dict_mean_to_impute_col[sub_ind] = temp_means.to_dict()

                    else:
                        filled_sub_ind_values = data_new.loc[
                            index_to_fill, column
                        ].fillna(
                            data_temp.groupby("revenue_Bkt")[column].transform("mean")
                        )
                        data_new.loc[index_to_fill, column] = filled_sub_ind_values

                        temp_means = data_temp.groupby("revenue_Bkt")[column].mean()
                        temp_means.index = temp_means.index.astype(str)
                        dict_mean_to_impute_col[sub_ind] = temp_means.to_dict()

                else:
                    filled_sub_ind_values = data_new.loc[index_to_fill, column].fillna(
                        data_temp.groupby("revenue_Bkt")[column].transform("mean")
                    )
                    data_new.loc[index_to_fill, column] = filled_sub_ind_values

                    temp_means = data_temp.groupby("revenue_Bkt")[column].mean()
                    temp_means.index = temp_means.index.astype(str)
                    dict_mean_to_impute_col[sub_ind] = temp_means.to_dict()

            dict_mean_to_impute[column] = dict_mean_to_impute_col

        with open(f"{path_intermediary}dict_means.json", "w") as fp:
            json.dump(dict_mean_to_impute, fp)

    else:
        with open(f"{path_intermediary}dict_means.json", "r") as fp:
            df_means = json.load(fp)
        for column in columnlist:
            data_temp = data_new[data_new[column].isna()][
                [column, "gics_sub_ind", "revenue"]
            ]

            for sub_ind in data_temp.gics_sub_ind.unique():
                for i in data_temp[data_temp.gics_sub_ind == sub_ind].index:
                    count = 0
                    temp_revenue = data_new.loc[i, "revenue"]

                    for interval in df_means[column][sub_ind].keys():
                        borne_inf, borne_sup = float(interval.split(",")[0][1:]), float(
                            interval.split(",")[1][:-1]
                        )
                        if count == 0:
                            borne_inf = -np.inf
                        if count == 9:
                            borne_sup = np.inf
                        if borne_inf <= temp_revenue and temp_revenue <= borne_sup:
                            data_new.loc[i, column] = df_means[column][sub_ind][
                                interval
                            ]
                        count += 1

    return data_new


def encoding(df, path_intermediary, train, restricted_features):
    """
    This function encodes and processes the input dataframe 'df' as follows:
    1. Removes rows from 'df' where 'company_id' is equal to 'OMH.AX', 'NHH.MC' or '1101.HK'.
    2. Defines lists of columns to be log-transformed, filled with mean industry values, and filled with next year values.
    3. Applies the 'processingrawdata' function to 'df' with the 'ThousandList' columns and fills missing values with mean industry values.
    4. Fills missing values for the 'FillList' columns with the next available year's value.
    5. Applies the 'fillmeanindustry' function to fill missing values for the 'FillList' columns with the mean industry value.
    6. Applies the 'logtransform' function to 'df' with the 'LogList' columns.
    7. Resets the index of the resulting dataframe to start from 0.

    Parameters:
    - df (pandas DataFrame): The DataFrame to be processed and encoded.
    - path_intermediary (str): The path to intermediary files used in data processing.
    - train (bool): A boolean flag indicating whether the data is part of the training set.
    - restricted_features (bool): A boolean flag indicating whether restricted features are used in the model.

    Returns:
    - df (pandas DataFrame): The encoded and processed DataFrame ready for modeling.

    """
    if restricted_features:
        FillList = ["asset", "ebit"]
        LogList = [
            "revenue",
            "asset",
            "cf1",
            "cf2",
            "cf3",
            "cf123",
            "ebit",
        ]
    else:
        LogList = [
            "revenue",
            "capex",
            "gppe",
            "nppe",
            "accu_dep",
            "intan",
            "cogs",
            "employees",
            "asset",
            "lt_debt",
            "cf1",
            "cf2",
            "cf3",
            "cf123",
            "energy_consumed",
            "energy_produced",
            "ebit",
            "ebitda",
        ]
        FillList = [
            # "revenue",
            "employees",
            "asset",
            "nppe",
            "intan",
            "capex",
            "age",
            "cap_inten",
            "gmar",
            "leverage",
            "energy_consumed",
            "energy_produced",
            "lt_debt",
            "gppe",
            "accu_dep",
            "cogs",
            "ebit",
            "ebitda",
        ]

    df = processingrawdata(df, restricted_features, train=train)
    df = fillmeanindustry(
        df,
        FillList,
        path_intermediary,
        train=train,
    )
    df = logtransform(df, LogList, path_intermediary, train=train)

    df.columns = df.columns.str.replace(r"\W", "_")
    return df


def selected_features(
    target,df_train, df_test, path_intermediary, extended_features, selec_sect
):
    """
    Selects a set of features from a pandas DataFrame.

    Parameters:
    - df (pandas.DataFrame): A pandas DataFrame containing the data.

    Returns:
    - list: A list of selected features as strings.
    """
    df = pd.concat([df_train, df_test])

    fixed_features = extended_features 

    for sect in selec_sect:
        fixed_features += [x for x in df.columns if sect + "_" in str(x)]

    # fixed_features = [f.replace(" ", "_") for f in fixed_features]

    pd.DataFrame(fixed_features, columns=["features"]).to_csv(
        path_intermediary + f"features_{target}.csv", index=False
    )

    return fixed_features


def outliers_preprocess(
    df,
    scope,
    threshold_over=2.5,
    threshold_under=1.5,
):
    """
    Filters out rows in where the target values are outliers with respect to subsectorial intensity.

    Parameters:
    - df: dataframe to be processed.

    Returns:

    - data_new: Processed dataframe with outlier rows removed.
    """
    index_to_drop = []

    df[f"intensity_{scope}"] = np.log(df[scope] / df["revenue"])

    for subindustry in df["gics_sub_ind"].unique():
        subset = df[df["gics_sub_ind"] == subindustry]
        if len(subset) > 10:
            median_subind = subset[f"intensity_{scope}"].quantile(0.50)
            Q1 = subset[f"intensity_{scope}"].quantile(0.25)
            Q3 = subset[f"intensity_{scope}"].quantile(0.75)
            IQR = Q3 - Q1

            max_subind = median_subind + threshold_over * IQR
            min_subind = median_subind - threshold_under * IQR

            condition = (subset[f"intensity_{scope}"] > max_subind) | (
                subset[f"intensity_{scope}"] < min_subind
            )

            if any(condition):
                index_to_drop.extend(subset.loc[condition].index.tolist())

    df = df[~(df.index.isin(index_to_drop))]
    return df


def custom_train_split(
    dataset,
    path_benchmark,
    path_intermediary,
    target,
    extended_features,
    restricted_features,
    selec_sect,
):
    """
    This function performs a custom train-test split on a given dataset, preprocesses the data, selects relevant features, and returns the processed data and relevant features.

    The input dataset is split into a train and test set using a benchmark date.
    Outliers are removed from the training set and mean imputation is performed on both the train and test set.
    Relevant features are selected using a predefined method.
    The processed data is split into features and target for both the train and test sets.
    The function returns the processed train and test data, relevant features, and the original test set with the selected features.
    
    Parameters:
    - dataset: The input dataset to be split and preprocessed.
    - path_benchmark: The path of a file containing the benchmark date for the train-test split.
    - path_intermediary: The path of a file containing the mean and minimum values for mean imputation.
    
    Returns:
    - X_train: The processed feature matrix for the training set.
    - y_train: The processed target variable for the training set.
    - X_test: The processed feature matrix for the test set.
    - y_test: The processed target variable for the test set.
    - features: The selected relevant features.
    - df_test: The original test set with the selected features.

    """
    try:
        
        df_train = pd.read_parquet(path_intermediary + f"df_train_{target}.parquet")
        df_test = pd.read_parquet(path_intermediary + f"df_test_{target}.parquet")

        features = pd.read_csv(path_intermediary + f"features_{target}.csv")["features"].tolist()
        print("Using pre created preprocessed files")

    except FileNotFoundError:
        print("Files not found, constructing them")

        df_train_before_imputation, df_test_before_imputation = df_split(
            dataset, path_benchmark
        )

        df_train, df_test = (
            encoding(
                
                df_train_before_imputation,
                path_intermediary,
                train=True,
                restricted_features=restricted_features,
            ),
            encoding(
                
                df_test_before_imputation,
                path_intermediary,
                train=False,
                restricted_features=restricted_features,
            ),
        )

        features = selected_features(
            target,
            df_train,
            df_test,
            path_intermediary,
            extended_features=extended_features,
            selec_sect=selec_sect,
        )
        df_train, df_test = set_columns(df_train, features), set_columns(
            df_test, features
        )

        df_train.index.name = "saved_index"
        df_test.index.name = "saved_index"

        df_train.to_parquet(path_intermediary + f"df_train_{target}.parquet")
        df_test.to_parquet(path_intermediary + f"df_test_{target}.parquet")

    df_train, df_test = target_preprocessing(df_train, target), target_preprocessing(
        df_test, target
    )
    if target in ["cf1", "cf3", "cf2", "cf123"]:
        # df_train = outliers_preprocess(df_train, target, threshold_under=threshold_under, threshold_over=threshold_over)
        X_train, y_train = df_train[features], df_train[target+"_log"]
        X_test, y_test = df_test[features], df_test[target+"_log"]

    elif target in ["cf1_cf123", "cf3_cf123", "cf2_cf123"]:
        target = target[:-6]
        # df_train = outliers_preprocess(df_train, target, threshold_under=threshold_under, threshold_over=threshold_over)
        X_train, y_train = df_train[features], df_train[target+"_log"]
        X_test, y_test = df_test[features], df_test[target+"_log"]
    else:
        print("unexpected target name, error")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        df_test,
        df_train
    )
