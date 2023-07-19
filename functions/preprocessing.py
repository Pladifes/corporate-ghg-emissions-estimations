import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import warnings
import json

warnings.simplefilter(action="ignore", category=FutureWarning)


def target_preprocessing(df, target):
    """
    A function to perform a target specific preprocessing on a pandas DataFrame.

    Args:
    df (pandas.DataFrame): A pandas DataFrame containing the data.
    target (str): The target variable to be used in the analysis.

    Returns:
    pandas.DataFrame: A preprocessed DataFrame.
    """
    df = df[df["FiscalYear"] >= 2011]
    df = df.dropna(subset=["CF1","CF2","CF3","CF123"])
    # if target == "CF1_log":
    #     df = df[df["FiscalYear"] >= 2005]
    #     df = df.dropna(subset=["CF1"])
    # elif target == "CF2_log":
    #     df = df[df["FiscalYear"] >= 2005]
    #     df = df.dropna(subset=["CF2"])
    # elif target == "CF3_log":
    #     df = df[df["FiscalYear"] >= 2011]
    #     df = df.dropna(subset=["CF3"])
    # elif target == "CF123_log":
    #     df = df[df["FiscalYear"] >= 2011]
    #     df = df.dropna(subset=["CF123"])
    return df


def df_split(df, path_Benchmark):
    """
    This function takes a pandas DataFrame and splits it into two parts based on whether the "Name" column of each row is in a benchmark list of company names.

    Parameters:
        - df (pandas DataFrame): The DataFrame to split.

    Returns:
        - tuple of pandas DataFrames: A tuple containing two DataFrames. The first DataFrame contains rows from the input DataFrame that do not have a "Name" value in the benchmark list, and the second DataFrame contains rows from the input DataFrame that do have a "Name" value in the benchmark list.
    """

    benchmark = pd.read_csv(path_Benchmark + "lst_companies_test_GICS.csv")
    benchmark_lst = benchmark["Name"].tolist()
    mask = df["Name"].isin(benchmark_lst)
    df_test = df[mask]
    df_train = df[~mask]
    return df_train, df_test


def set_columns(df, features):
    """
    Align the column names of two pandas DataFrames and replace special characters in column names with underscores.

    Args:
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


def logtransform(df, ls, target, path_results, train):
    """
    This function calculates the base-10 logarithm of the values of keys in the given dictionary for the corresponding keys in the given list and returns the updated dictionary.

    Parameters:
    res (dict): The dictionary in which the logarithm needs to be calculated.
    ls (list): The list of keys whose corresponding values need to be logged.
    target (str): The target column name.
    path_results (str): The path to save the output csv file.
    train (bool, optional): Flag indicating if the function is being used for training or testing. Default is True.

    Returns:
    dict: The dictionary with updated values.
    """
    res = df.copy()
    columns_min = []

    if train:
        for l in ls:
            if l in ["CF1", "CF3", "CF2", "CF123"]:
                res[l + "_log"] = np.log10(res[l] + 1)
            else:
                res[l + "_log"] = np.log10(res[l] - res[l].min() + 1)
            columns_min_dict = {
                # "target": target,
                "column": l,
                "min_value": res[l].min(),
            }
            columns_min.append(columns_min_dict)

        columns_min_df = pd.DataFrame(columns_min)
        columns_min_df.to_csv(path_results + f"{target}_columns_min.csv", index=False)

    else:
        columns_min = pd.read_csv(path_results + f"{target}_columns_min.csv")
        for l in ls:
            if l in ["CF1", "CF3", "CF2", "CF123"]:
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


def label_FinalCO2Law(row):
    """
    This function applies a label to a row in a pandas DataFrame based on the values in the "CO2Law", "CO2Status", and "CO2Coverage" columns.

    Parameters:
    row (pandas.Series): The row to be labeled.

    Returns:
    str: The label for the row.

    """
    if row["CO2Law"] == "No":
        return "No CO2 Law"
    if row["CO2Law"] == "TBD":
        return "No CO2 Law"
    if row["CO2Law"] == "Yes" and row["CO2Status"] == "Scheduled":
        return "No CO2 Law"
    if (
        row["CO2Law"] == "Yes"
        and row["CO2Status"] == "Implemented"
        and row["CO2Coverage"] == "Subnational"
    ):
        return "Subnational Implemented"
    if (
        row["CO2Law"] == "Yes"
        and row["CO2Status"] == "Implemented"
        and row["CO2Coverage"] == "National"
    ):
        return "National Implemented"
    if (
        row["CO2Law"] == "Yes"
        and row["CO2Status"] == "Implemented"
        and row["CO2Coverage"] == "Regional"
    ):
        return "Regional Implemented"
    else:
        return "Other"


# def processingrawdata(data_old, open_data):
#     """
#     Process raw data by applying various transformations and generating dummy variables.

#     Parameters:
#     - data_old (pandas DataFrame): the raw data to be processed.
#     - ThousandList (list of strings): a list of column names for which the values should be multiplied by 1000.
#     - fillGMAR (bool, default False): if True, fill any missing values in the "GMAR" column with 1.

#     Returns:
#     - data_new (pandas DataFrame): the processed data, with new columns generated for dummy variables.
#     """
#     data_new = data_old.copy()
#     data_new["FinalCO2Law"] = data_new.apply(lambda row: label_FinalCO2Law(row), axis=1)

    # if not open_data:
    #     data_new["GICSGroup"] = data_new["GICSGroup"].astype(str)
    #     data_new["GICSSector"] = data_new["GICSSector"].astype(str)
    #     data_new["GICSInd"] = data_new["GICSInd"].astype(str)

    # data_new["GICSSubInd"] = data_new["GICSSubInd"].astype(str)
    # data_new = pd.concat(
    #     [data_new, pd.get_dummies(data_new["IncomeGroup"], prefix="IncomeGroup_")],
    #     axis=1,
    # )
    # data_new = pd.concat(
    #     [data_new, pd.get_dummies(data_new["FinalCO2Law"], prefix="FinalCO2Law_")],
    #     axis=1,
    # )
    # data_new = pd.concat(
    #     [data_new, pd.get_dummies(data_new["FiscalYear"], prefix="FiscalYear_")], axis=1
    # )

    # if not open_data:
    #     data_new = pd.concat(
    #         [data_new, pd.get_dummies(data_new["GICSSector"], prefix="GICSSector_")],
    #         axis=1,
    #     )
    #     data_new = pd.concat(
    #         [data_new, pd.get_dummies(data_new["GICSGroup"], prefix="GICSGroup_")],
    #         axis=1,
    #     )
    #     data_new = pd.concat(
    #         [data_new, pd.get_dummies(data_new["GICSInd"], prefix="GICSInd_")], axis=1
    #     )

    # data_new = pd.concat(
    #     [data_new, pd.get_dummies(data_new["GICSSubInd"], prefix="GICSSubInd_")], axis=1
    # )

    # return data_new


def processingrawdata(data_old, open_data):
    """
    Process raw data by applying various transformations and generating dummy variables.

    Parameters:
    - data_old (pandas DataFrame): the raw data to be processed.
    - open_data (bool): if True, the data is open data.

    Returns:
    - data_new (pandas DataFrame): the processed data, with new columns generated for ordinal encoded variables.
    """
    data_new = data_old.copy()
    data_new["FinalCO2Law"] = data_new.apply(lambda row: label_FinalCO2Law(row), axis=1)

    if not open_data:
        data_new["GICSGroup"] = data_new["GICSGroup"].astype(str)
        data_new["GICSSector"] = data_new["GICSSector"].astype(str)
        data_new["GICSInd"] = data_new["GICSInd"].astype(str)

    data_new["GICSSubInd"] = data_new["GICSSubInd"].astype(str)

    income_group_encoder = OrdinalEncoder()
    data_new["IncomeGroup_encoded"] = income_group_encoder.fit_transform(data_new[["IncomeGroup"]])

    final_co2_law_encoder = OrdinalEncoder()
    data_new["FinalCO2Law_encoded"] = final_co2_law_encoder.fit_transform(data_new[["FinalCO2Law"]])

    fiscal_year_encoder = OrdinalEncoder()
    data_new["FiscalYear_encoded"] = fiscal_year_encoder.fit_transform(data_new[["FiscalYear"]])

    if not open_data:
        data_new = pd.concat(
            [data_new, pd.get_dummies(data_new["GICSSector"], prefix="GICSSector_")],
            axis=1,
        )
        data_new = pd.concat(
            [data_new, pd.get_dummies(data_new["GICSGroup"], prefix="GICSGroup_")],
            axis=1,
        )
        data_new = pd.concat(
            [data_new, pd.get_dummies(data_new["GICSInd"], prefix="GICSInd_")], axis=1
        )

    data_new = pd.concat(
        [data_new, pd.get_dummies(data_new["GICSSubInd"], prefix="GICSSubInd_")], axis=1
    )

    return data_new



def fillmeanindustry(
    data_old, groupvalue, columnlist, target, path_intermediary, train, old_pipe
):
    """
    Fill in missing values in a pandas DataFrame by using the mean of the corresponding group values.

    Parameters:
    - data_old (pandas DataFrame): the data to be processed.
    - groupvalue (str): the column name of the grouping variable.
    - columnlist (list of strings): a list of column names for which missing values should be filled.

    Returns:
    - data_new (pandas DataFrame): the processed data with missing values filled in.
    """
    data_new = data_old.copy()
    if train:
        if old_pipe:

            for column in columnlist:
                data_new[column] = data_new[column].fillna(
                    data_new.groupby(groupvalue)[column].transform("mean")
                )
            columns_mean_df = data_new.groupby(groupvalue)[columnlist].mean()
            columns_mean_df[columnlist].to_csv(
                f"{path_intermediary}/{target}_columns_mean.csv", index=True
            )

        else:

            nb_per_sub_ind = data_new.groupby(["GICSSubInd"])[columnlist].count()
            dict_mean_to_impute = pd.DataFrame(columns=columnlist).to_dict()

            for column in columnlist:
                dict_mean_to_impute_col = {}

                for sub_ind in nb_per_sub_ind[column].index:
                    index_to_fill = data_new[data_new.GICSSubInd == sub_ind].index
                    data_temp = data_new[data_new.GICSSubInd == sub_ind][
                        [column, "Revenue"]
                    ]
                    data_temp["Revenue_Bkt"] = pd.qcut(
                        data_temp.Revenue, 10, duplicates="drop"
                    )

                    if data_temp.groupby("Revenue_Bkt").count()[column].min() < 10:
                        ind = sub_ind[:-4] + sub_ind[-2:]
                        data_temp = data_new[data_new.GICSInd == ind][
                            [column, "Revenue"]
                        ]
                        data_temp["Revenue_Bkt"] = pd.qcut(
                            data_temp.Revenue, 10, duplicates="drop"
                        )

                        if data_temp.groupby("Revenue_Bkt").count()[column].min() < 10:
                            grp = ind[:-4] + ind[-2:]
                            data_temp = data_new[data_new.GICSGroup == grp][
                                [column, "Revenue"]
                            ]
                            data_temp["Revenue_Bkt"] = pd.qcut(
                                data_temp.Revenue, 10, duplicates="drop"
                            )

                            if (
                                data_temp.groupby("Revenue_Bkt").count()[column].min()
                                < 10
                            ):
                                sect = grp[:-4] + grp[-2:]
                                data_temp = data_new[data_new.GICSSector == sect][
                                    [column, "Revenue"]
                                ]
                                data_temp["Revenue_Bkt"] = pd.qcut(
                                    data_temp.Revenue, 10, duplicates="drop"
                                )
                                filled_sub_ind_values = data_new.loc[
                                    index_to_fill, column
                                ].fillna(
                                    data_temp.groupby("Revenue_Bkt")[column].transform(
                                        "mean"
                                    )
                                )
                                data_new.loc[
                                    index_to_fill, column
                                ] = filled_sub_ind_values

                                temp_means = data_temp.groupby("Revenue_Bkt")[
                                    column
                                ].mean()
                                temp_means.index = temp_means.index.astype(str)
                                dict_mean_to_impute_col[sub_ind] = temp_means.to_dict()

                            else:
                                filled_sub_ind_values = data_new.loc[
                                    index_to_fill, column
                                ].fillna(
                                    data_temp.groupby("Revenue_Bkt")[column].transform(
                                        "mean"
                                    )
                                )
                                data_new.loc[
                                    index_to_fill, column
                                ] = filled_sub_ind_values

                                temp_means = data_temp.groupby("Revenue_Bkt")[
                                    column
                                ].mean()
                                temp_means.index = temp_means.index.astype(str)
                                dict_mean_to_impute_col[sub_ind] = temp_means.to_dict()

                        else:

                            filled_sub_ind_values = data_new.loc[
                                index_to_fill, column
                            ].fillna(
                                data_temp.groupby("Revenue_Bkt")[column].transform(
                                    "mean"
                                )
                            )
                            data_new.loc[index_to_fill, column] = filled_sub_ind_values

                            temp_means = data_temp.groupby("Revenue_Bkt")[column].mean()
                            temp_means.index = temp_means.index.astype(str)
                            dict_mean_to_impute_col[sub_ind] = temp_means.to_dict()

                    else:

                        filled_sub_ind_values = data_new.loc[
                            index_to_fill, column
                        ].fillna(
                            data_temp.groupby("Revenue_Bkt")[column].transform("mean")
                        )
                        data_new.loc[index_to_fill, column] = filled_sub_ind_values

                        temp_means = data_temp.groupby("Revenue_Bkt")[column].mean()
                        temp_means.index = temp_means.index.astype(str)
                        dict_mean_to_impute_col[sub_ind] = temp_means.to_dict()

                dict_mean_to_impute[column] = dict_mean_to_impute_col

            with open(f"{path_intermediary}/{target}_dict_means.json", "w") as fp:
                json.dump(dict_mean_to_impute, fp)

    else:  # test/pro :
        if old_pipe:
            df_means = pd.read_csv(f"{path_intermediary}/{target}_columns_mean.csv")
            df_means = df_means.set_index(groupvalue)

            for column in columnlist:
                means_column = df_means[column]
                means_column = means_column.dropna()
                # if mean_column < data_new[column].mean():
                #     data_new[column] = data_new[column].fillna(
                #         data_new.groupby(groupvalue)[column].transform(
                #             lambda x: x.fillna(mean_column)
                #         )
                #     )
                # else:

                for keys in means_column.index[:1]:
                    curr_index = data_new[
                        (data_new[groupvalue[0]].astype(float) == keys[0])
                        & (data_new[groupvalue[1]].astype(float) == keys[1])
                    ].index
                    data_new.loc[curr_index, column] = data_new.loc[
                        curr_index, column
                    ].fillna(means_column[keys])

        else:
            with open(f"{path_intermediary}/{target}_dict_means.json", "r") as fp:
                df_means = json.load(fp)

            for column in columnlist:
                data_temp = data_new[data_new[column].isna()][
                    [column, "GICSSubInd", "Revenue"]
                ]

                for (
                    sub_ind
                ) in (
                    data_temp.GICSSubInd.unique()
                ):  # loop on sub ind that have na for the current column

                    for i in data_temp[data_temp.GICSSubInd == sub_ind].index:
                        count = 0
                        temp_revenue = data_new.loc[i, "Revenue"]

                        for interval in df_means[column][sub_ind].keys():
                            # print(interval)
                            borne_inf, borne_sup = float(
                                interval.split(",")[0][1:]
                            ), float(interval.split(",")[1][:-1])
                            if count == 0:
                                borne_inf = -np.inf
                            if count == 9:
                                borne_sup = np.inf
                            if borne_inf <= temp_revenue and temp_revenue <= borne_sup:
                                data_new.loc[i, column] = df_means[column][sub_ind][
                                    interval
                                ]  # impute saved mean
                            count += 1

    return data_new


def fillnextyear(data_old, groupvalue, columnlist):
    """
    Fills in missing values in a pandas DataFrame by using the 'backfill' method,
    within groups defined by a given column or columns.

    Parameters:
    data_old:pandas DataFrame
        The input DataFrame with missing values to be filled.
    groupvalue:str or list of str
        The name(s) of the column(s) to be used as group keys.
    columnlist:str or list of str
        The name(s) of the column(s) to be filled.

    Returns:
    data_new:pandas DataFrame
        The output DataFrame with missing values filled, using the 'backfill'
        method within groups defined by the given column(s).
    """
    data_new = data_old.sort_values(by=["FiscalYear"])
    for column in columnlist:
        data_new[column] = data_new.groupby(groupvalue)[column].fillna(
            method="bfill", limit=1
        )
    return data_new


def encoding(df, target, path_intermediary, train, fill_grp, old_pipe, open_data):
    """
    This function encodes and processes the input dataframe 'df' as follows:
    1. Removes rows from 'df' where 'FinalEikonID' is equal to 'OMH.AX', 'NHH.MC' or '1101.HK'.
    2. Defines lists of columns to be log-transformed, filled with mean industry values, and filled with next year values.
    3. Applies the 'processingrawdata' function to 'df' with the 'ThousandList' columns and fills missing values with mean industry values.
    4. Fills missing values for the 'FillList' columns with the next available year's value.
    5. Applies the 'fillmeanindustry' function to fill missing values for the 'FillList' columns with the mean industry value.
    6. Applies the 'logtransform' function to 'df' with the 'LogList' columns.
    7. Resets the index of the resulting dataframe to start from 0.

    Parameters:
    -----------
    df:pandas.DataFrame
        Input dataframe to be processed and encoded.

    Returns:
    --------
    df:pandas.DataFrame
        Encoded and processed dataframe.
    """
    if open_data:
        FillList = ["Asset", "EBIT"]
        LogList = [
            "Revenue",
            "Asset",
            "CF1",
            "CF2",
            "CF3",
            "CF123",
            "EBIT",
        ]
    else:
        FillList = [
            # "Revenue",
            "EMP",
            "Asset",
            "NPPE",
            "INTAN",
            "CapEx",
            "Age",
            "CapInten",
            "GMAR",
            "Leverage",
            "ENEConsume",
            "ENEProduce",
            "LTDebt",
            "GPPE",
            "AccuDep",
            "COGS",
            "EBIT",
            "EBITDA",
        ]

        LogList = [
            "Revenue",
            "CapEx",
            "GPPE",
            "NPPE",
            "AccuDep",
            "INTAN",
            "COGS",
            "EMP",
            "Asset",
            "LTDebt",
            "CF1",
            "CF2",
            "CF3",
            "CF12",
            "CF123",
            "ENEConsume",
            "ENEProduce",
            "EBIT",
            "EBITDA",
        ]

    df = processingrawdata(df, open_data)
    # if train:
    #     missing_values_sum = pd.DataFrame(
    #         df[FillList].isnull().sum(),
    #         columns=["Missing Values train before imputation"],
    #     )

    # df = fillnextyear(df, "Name", FillList)
    # if train:
    #     missing_values_sum["Missing Values train after imputation"] = (
    #         df[FillList].isnull().sum()
    #     )
    #     missing_values_sum.to_csv(f"missing_values_sum_ffill_{target}.csv")

    df = fillmeanindustry(
        df,
        [fill_grp, "FiscalYear"],
        FillList,
        target,
        path_intermediary,
        train=train,
        old_pipe=old_pipe,
    )

    df = logtransform(df, LogList, target, path_intermediary, train=train)

    df.columns = df.columns.str.replace(r"\W", "_")
    df = df.reset_index(drop=True)

    return df


# def selected_features(
#     df_train, df_test, target, path_intermediary, extended_features, selec_sect
# ):
#     """
#     Selects a set of features from a pandas DataFrame.

#     Args:
#         df (pandas.DataFrame): A pandas DataFrame containing the data.
#         target (str): The name of the target variable.

#     Returns:
#         list: A list of selected features as strings.
#     """

#     if target == "CF1_log":
#         df = pd.concat([df_train, df_test])
#         with_co2_law_dummy = [x for x in df.columns if "FinalCO2Law_" in str(x)]
#         with_income_group_dummy = [x for x in df.columns if "IncomeGroup_" in str(x)]
#         with_fiscal_year_dummy = [x for x in df.columns if "FiscalYear_" in str(x)]

#         fixed_features = (
#             extended_features
#             + with_fiscal_year_dummy
#             + with_income_group_dummy
#             + with_co2_law_dummy
#         )

#         for sect in selec_sect:
#             fixed_features += [x for x in df.columns if sect + "__" in str(x)]

#         fixed_features = [f.replace(" ", "_") for f in fixed_features]

#         pd.DataFrame(fixed_features, columns=["features"]).to_csv(
#             path_intermediary + "features.csv", index=False
#         )

#     else:
#         fixed_features = pd.read_csv(path_intermediary + "/features.csv")[
#             "features"
#         ].tolist()

#     return fixed_features

def selected_features(
    df_train, df_test, target, path_intermediary, extended_features, selec_sect
):
    """
    Selects a set of features from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): A pandas DataFrame containing the data.
        target (str): The name of the target variable.

    Returns:
        list: A list of selected features as strings.
    """

    if target == "CF1_log":
        df = pd.concat([df_train, df_test])

        encoded_variables = ["FinalCO2Law_encoded","IncomeGroup_encoded","FiscalYear_encoded"]
        fixed_features = (
            extended_features
            + encoded_variables
        )

        for sect in selec_sect:
            fixed_features += [x for x in df.columns if sect + "__" in str(x)]

        fixed_features = [f.replace(" ", "_") for f in fixed_features]

        pd.DataFrame(fixed_features, columns=["features"]).to_csv(
            path_intermediary + "features.csv", index=False
        )

    else:
        fixed_features = pd.read_csv(path_intermediary + "/features.csv")[
            "features"
        ].tolist()

    return fixed_features



def compagnies_outlier(
    Refinitiv_data,
    scope,
    nb_std,
):
    company_names_list = []
    selected_companies = []

    Refinitiv_data[f"intensity_{scope}"] = (
        Refinitiv_data[scope] / Refinitiv_data["Revenue"]
    )

    for subindustry in Refinitiv_data["GICSName"].unique():
        subset = Refinitiv_data[Refinitiv_data["GICSName"] == subindustry]

        std_subind = np.std(subset[f"intensity_{scope}"])
        mean_subind = np.mean(subset[f"intensity_{scope}"])
        max_subind = mean_subind + nb_std * std_subind
        min_subind = mean_subind - nb_std * std_subind

        condition = (subset[f"intensity_{scope}"] > max_subind) | (
            subset[f"intensity_{scope}"] < min_subind
        )

        if any(condition):
            selected_companies.extend(
                subset.loc[condition, ["Name", "FiscalYear"]].apply(
                    lambda x: (x["Name"], x["FiscalYear"]), axis=1
                )
            )

    selected_companies_years = selected_companies
    company_names = [company[0] for company in selected_companies_years]

    company_names_list.append(company_names)
    return company_names_list


def outliers_preprocess(df, target, low, high):
    """
    This function filters out rows in where the target values are below/above the low/high quantile.

    Parameters:
    -----------
    df:pandas.DataFrame
        Input dataframe to be processed.

    Returns:
    --------
    data_new:pandas.DataFrame
        Processed dataframe with outlier rows removed.
    """
    quantiles = df.quantile([low, high])
    if low != 0:
        df = df[df[target] > quantiles.loc[low, target]]
    if high != 1:
        df = df[df[target] < quantiles.loc[high, target]]

    # company_names_list = compagnies_outlier(df, target, 2)
    # df = df[~df["ISIN"].isin(company_names_list)]
    return df


def custom_train_split(
    dataset,
    path_Benchmark,
    path_intermediary,
    target,
    low=0.01,
    high=1,
    extended_features=[
        "Revenue_log",
        "EMP_log",
        "Asset_log",
        "NPPE_log",
        "CapEx_log",
        "Age",
        "CapInten",
        "GMAR",
        "Leverage",
        "Price",
        "FuelIntensity",
        "FiscalYear",
        "ENEConsume_log",
        "ENEProduce_log",
        "INTAN_log",
        "GPPE_log",
        "EBIT_log",
        "EBITDA_log",
        "AccuDep_log",
        "COGS_log",
        "LTDebt_log",
    ],
    fill_grp="GICSGroup",
    old_pipe=True,
    open_data=False,
    selec_sect=["GICSSect", "GICSGroup", "GICSInd", "GICSSubInd"],
):
    """
    This function performs a custom train-test split on a given dataset, preprocesses the data, selects relevant features, and returns the processed data and relevant features.

    The input dataset is split into a train and test set using a benchmark date.
    Outliers are removed from the training set and mean imputation is performed on both the train and test set.
    Relevant features are selected using a predefined method.
    The processed data is split into features and target for both the train and test sets.
    The function returns the processed train and test data, relevant features, and the original test set with the selected features.
    Parameters:

    dataset: The input dataset to be split and preprocessed.
    path_Benchmark: The path of a file containing the benchmark date for the train-test split.
    target: The name of the target variable.
    path_intermediary: The path of a file containing the mean and minimum values for mean imputation.
    Returns:

    X_train: The processed feature matrix for the training set.
    y_train: The processed target variable for the training set.
    X_test: The processed feature matrix for the test set.
    y_test: The processed target variable for the test set.
    features: The selected relevant features.
    df_test: The original test set with the selected features.

    """
    df_train_before_imputation, df_test_before_imputation = df_split(
        dataset, path_Benchmark
    )
    df_train, df_test = (
        encoding(
            df_train_before_imputation,
            target,
            path_intermediary,
            train=True,
            fill_grp=fill_grp,
            old_pipe=old_pipe,
            open_data=open_data,
        ),
        encoding(
            df_test_before_imputation,
            target,
            path_intermediary,
            train=False,
            fill_grp=fill_grp,
            old_pipe=old_pipe,
            open_data=open_data,
        ),
    )
    df_train = outliers_preprocess(df_train, target, low=low, high=high)
    features = selected_features(
        df_train,
        df_test,
        target,
        path_intermediary,
        extended_features=extended_features,
        selec_sect=selec_sect,
    )
    df_train, df_test = set_columns(df_train, features), set_columns(df_test, features)
    X_train, y_train = df_train[features], df_train[target]
    X_test, y_test = df_test[features], df_test[target]
    return (
        X_train,
        y_train,
        X_test,
        y_test,
        df_test,
        df_train,
        df_test_before_imputation,
        df_train_before_imputation,
    )
