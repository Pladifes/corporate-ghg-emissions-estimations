import pickle as pkl
import pandas as pd
import numpy as np


from functions.merged_dataset_creation import merge_datasets
from functions.preprocessing import encoding, set_columns


def apply_model_on_forbes_data(
    path_rawdata="data/raw_data/",
    path_results="results/",
    path_intermediary="data/intermediary_data/",
    path_models="models/",
    save=False,
):
    """
    Apply pre-saved machine learning models to Forbes data to predict scope 1, 2, and 3 emissions.

    Parameters:
        path_rawdata (str, optional): Path to the directory containing raw data files. Defaults to "data/raw_data/".
        path_results (str, optional): Path to the directory where the results will be saved. Defaults to "results/".
        path_intermediary (str, optional): Path to the directory containing intermediary data. Defaults to "data/intermediary_data/".
        path_models (str, optional): Path to the directory containing pre-saved machine learning models. Defaults to "models/".
        save (bool, optional): If True, save the results to an Excel file. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing Forbes data with additional columns for estimated emissions.

    """
    df_forbes = pd.read_excel(path_rawdata + "forbes_2007_2022_completed.xlsx")
    carbon_pricing = pd.read_csv(
        path_rawdata + "carbon_pricing_preprocessed_2023.csv",
    )
    income_group = pd.read_csv(
        path_rawdata + "income_group_preprocessed_2023.csv",
    )
    fuel_intensity = pd.read_csv(path_rawdata + "fuel_mix_2023.csv")
    region_mapping = pd.read_excel(path_rawdata + "country_region_mapping.xlsx")

    del df_forbes["market_value"]
    del df_forbes["unique_id"]
    del df_forbes["forbes_year"]

    df_forbes = df_forbes.rename(
        columns={
            "country": "country_hq",
            "industry": "gics_sub_ind",
            "sales": "revenue",
            "profits": "ebit",
            "assets": "asset",
        }
    )
    df_forbes["fiscal_year"] = df_forbes.fiscal_year.astype(int)

    df_forbes_merged = merge_datasets(
        df_forbes,
        carbon_pricing,
        income_group,
        fuel_intensity,
        region_mapping,
    )

    df_forbes_merged["fuel_intensity"] = df_forbes_merged.fuel_intensity.fillna(
        df_forbes_merged.fuel_intensity.median()
    )

    df_forbes_merged["cf1"] = np.ones(len(df_forbes_merged))
    df_forbes_merged["cf2"] = np.ones(len(df_forbes_merged))
    df_forbes_merged["cf3"] = np.ones(len(df_forbes_merged))
    df_forbes_merged["cf123"] = np.ones(len(df_forbes_merged))

    for scope in ["cf1", "cf2", "cf3", "cf123"]:
        dataset = encoding(
            df_forbes_merged,
            path_intermediary,
            train=False,
            restricted_features=True,
        )
        features = pd.read_csv(path_intermediary + "features.csv").squeeze().tolist()
        dataset = set_columns(dataset, features)
        dataset = dataset[features]
        reg = pkl.load(open(path_models + "{}_log_model.pkl".format(scope), "rb"))
        scope_pred = reg.predict(dataset)
        df_forbes[scope + "_e"] = np.power(10, scope_pred + 1)

    df_forbes["cf1_e + cf2_e + cf3_e"] = (
        df_forbes["cf1_e"]
        + df_forbes["cf2_e"]
        + df_forbes["cf3_e"]
        + df_forbes["cf123_e"]
    )

    if save:
        df_forbes.to_excel(
            path_results + "pladifes_free_emissions_estimates.xlsx", index=False
        )

    return df_forbes


def apply_model_on_raw_data(
    dataset,
    path_intermediary="data/intermediary_data",
    path_models="models/",
    path_results="results/",
    save=False,
):
    """
    Apply pre-saved models to raw data to predict scope 1, 2, and 3 emissions.

    Parameters:
    - dataset (DataFrame): The raw data DataFrame to which the models will be applied.
    - path_intermediary (str): The path to the intermediary data directory.
    - path_models (str): The path to the directory containing pre-trained models.
    - path_results (str): The path to the results directory where the output will be saved.
    - save (bool): Whether to save the emission estimates as an Excel file.

    Returns:
    - estimations (DataFrame): A DataFrame containing emissions estimates for the input dataset.

    This function performs the following steps:
    1. Initializes an empty DataFrame called 'estimations' to store the output.
    2. Iterates through scope levels (CF1, CF2, CF3, CF123) and applies pre-trained models to make emissions predictions.
    3. Calculates the sum of CF1, CF2, CF3, and CF123 emissions.
    4. Optionally saves the results to an Excel file if 'save' is set to True.

    Note: Make sure the models used are appropriate for the provided 'dataset'.

    """
    estimations = dataset[["company_id", "fiscal_year", "isin", "ticker", "gics_name"]]
    for scope in ["cf1", "cf2", "cf3", "cf123"]:
        preprocessed_dataset = encoding(
            dataset,
            path_intermediary,
            train=False,
            restricted_features=True,
        )
        features = pd.read_csv(path_intermediary + "features.csv").squeeze().tolist()
        preprocessed_dataset = set_columns(preprocessed_dataset, features)
        preprocessed_dataset = preprocessed_dataset[features]
        reg = pkl.load(open(path_models + "{}_log_model.pkl".format(scope), "rb"))
        scope_pred = reg.predict(preprocessed_dataset)
        estimations[scope + "_e"] = np.power(10, scope_pred + 1)

    estimations["cf1_e + cf2_e + cf3_e"] = (
        estimations["cf1_e"]
        + estimations["cf2_e"]
        + estimations["cf3_e"]
        + estimations["cf123_e"]
    )

    if save:
        estimations.to_excel(
            path_results + "pladifes_free_emissions_estimates.xlsx", index=False
        )

    return estimations
