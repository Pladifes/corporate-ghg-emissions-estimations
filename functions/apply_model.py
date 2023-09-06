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
    This function apply pre saved models to forbes data to predict scope 1, 2 and 3 emissions.
    WARINING : models has to be restricted to Sales (revenue), Profits (ebit) and assets (asset), and gics_sub_ind Names.
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
        df_forbes["cf1_e"] + df_forbes["cf2_e"] + df_forbes["cf3_e"] + df_forbes["cf123_e"]
    )

    if save:
        df_forbes.to_excel(path_results + "pladifes_free_emissions_estimates.xlsx", index=False)

    return df_forbes


def apply_model_on_raw_data(
    dataset, path_intermediary="data/intermediary_data", path_models="models/", path_results="results/", save=False
):
    """
    This function apply pre saved models to raw data to predict scope 1, 2 and 3 emissions.
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
        estimations["cf1_e"] + estimations["cf2_e"] + estimations["cf3_e"] + estimations["cf123_e"]
    )

    if save:
        estimations.to_excel(path_results + "pladifes_free_emissions_estimates.xlsx", index=False)

    return estimations
