import pickle as pkl
import pandas as pd
import numpy as np


from functions.merged_dataset_creation import (
    CarbonPricing_preprocess,
    IncomeGroup_preprocess,
    merge_datasets,
)
from functions.preprocessing import encoding, set_columns


def apply_model_on_forbes_data(
    path_rawdata="data/raw_data/",
    path_results="results/",
    path_intermediary="data/intermediary_data/",
    path_models="models/",
    save=False,
):
    """
    Apply pre-saved models to Forbes data to predict scope 1, 2, and 3 emissions.

    Parameters:
    - path_rawdata (str): The path to the raw data directory.
    - path_results (str): The path to the results directory where the output will be saved.
    - path_intermediary (str): The path to the intermediary data directory.
    - path_models (str): The path to the directory containing pre-trained models.
    - save (bool): Whether to save the results as an Excel file.

    WARNING: The models must be restricted to Sales (Revenue), Profits (EBIT), Assets (Asset), and GICSSubInd Names.

    Returns:
    - df_forbes (DataFrame): A DataFrame containing Forbes data with added emissions predictions.

    This function performs the following steps:
    1. Reads and preprocesses mapping data, CarbonPricing data, IncomeGroup data, and FuelIntensity data.
    2. Reads Forbes data, performs data cleaning, and adds a 'Region' column.
    3. Renames columns for consistency and converts the 'FiscalYear' column to integers.
    4. Merges datasets and handles missing values in the 'FuelIntensity' column.
    5. Applies pre-trained models to predict emissions (CF1, CF2, CF3, and CF123) and adds them to the Forbes data.
    6. Calculates the sum of CF1, CF2, CF3, and CF123 emissions.
    7. Optionally saves the results to an Excel file.

    Example usage:
    df = apply_model_on_forbes_data(save=True)
    """
    mapping = pd.read_excel(path_rawdata + "country_region_mapping.xlsx")
    mapping_dict = mapping.set_index("Country").to_dict()["Region"]

    CarbonPricing = pd.read_excel(
        path_rawdata + "Carbon Price Rework 20230405.xlsx",
    )
    IncomeGroup = pd.read_excel(
        path_rawdata + "updated_income_group.xlsx",
    )
    FuelIntensity = pd.read_csv(
        path_rawdata + "2021FuelMix.csv", encoding="latin-1"
    ).rename(columns={"Value": "FuelIntensity"})

    df_forbes = pd.read_excel(path_rawdata + "forbes_2007_2022_completed.xlsx")
    del df_forbes["Market Value"]
    del df_forbes["unique_id"]
    del df_forbes["Forbes Year"]

    df_forbes = df_forbes.drop_duplicates()
    df_forbes = df_forbes.dropna()
    df_forbes = df_forbes.reset_index(drop=True)
    df_forbes = df_forbes.replace(
        {
            "United States": "United States of America",
            "South Korea": "Korea; Republic (S. Korea)",
            "Ireland": "Ireland; Republic of",
            "Hong Kong/China": "China",
            "Australia/United Kingdom": "Australia",
            "Netherlands/United Kingdom": "Netherlands",
            "Panama/United Kingdom": "Panama",
            "Hong Kong-China": "Hong Kong",
            "North America": "United States of America",
            "South America": "Venezuela",
            "Health Care Equipment & Svcs": "United States of America",
            "Medical Equipment & Supplies": "United States of America",
            "Europe": "Spain",
            "United Kingdom/South Africa": "South Africa",
            "United Kingdom/Netherlands": "Netherlands",
            "United Kingdom/Australia": "Australia",
            "Canada/United Kingdom": "Canada",
        }
    )

    df_forbes["Region"] = df_forbes["Country"].apply(lambda x: mapping_dict[x])

    df_forbes = df_forbes.rename(
        columns={
            "Country": "CountryHQ",
            "Industry": "GICSSubInd",
            "Sales": "Revenue",
            "Profits": "EBIT",
            "Assets": "Asset",
        }
    )
    df_forbes["FiscalYear"] = df_forbes.FiscalYear.astype(int)

    CarbonPricing_Transposed = CarbonPricing_preprocess(CarbonPricing)
    IncomeGroup_Transposed = IncomeGroup_preprocess(IncomeGroup)

    df_forbes_merged = merge_datasets(
        df_forbes,
        CarbonPricing_Transposed,
        IncomeGroup_Transposed,
        FuelIntensity,
        CDP_preprocessed=None,
    )

    df_forbes_merged["FuelIntensity"] = df_forbes_merged.FuelIntensity.fillna(
        df_forbes_merged.FuelIntensity.median()
    )

    df_forbes_merged["CF1"] = np.ones(len(df_forbes_merged))
    df_forbes_merged["CF2"] = np.ones(len(df_forbes_merged))
    df_forbes_merged["CF3"] = np.ones(len(df_forbes_merged))
    df_forbes_merged["CF123"] = np.ones(len(df_forbes_merged))

    for scope in ["CF1", "CF2", "CF3", "CF123"]:
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
        df_forbes[scope + "_E"] = np.power(10, scope_pred + 1)

    df_forbes["CF1_E + CF2_E + CF3_E"] = (
        df_forbes["CF1_E"]
        + df_forbes["CF2_E"]
        + df_forbes["CF3_E"]
        + df_forbes["CF123_E"]
    )

    if save:
        df_forbes.to_excel(
            path_results + "Pladifes_free_emissions_estimates.xlsx", index=False
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

    Example usage:
    estimations = apply_model_on_raw_data(raw_data, save=True)
    """
    estimations = dataset[["FinalEikonID", "FiscalYear", "ISIN", "Ticker", "GICSName"]]
    for scope in ["CF1", "CF2", "CF3", "CF123"]:
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
        estimations[scope + "_E"] = np.power(10, scope_pred + 1)

    estimations["CF1_E + CF2_E + CF3_E"] = (
        estimations["CF1_E"]
        + estimations["CF2_E"]
        + estimations["CF3_E"]
        + estimations["CF123_E"]
    )

    if save:
        estimations.to_excel(
            path_results + "Pladifes_free_emissions_estimates.xlsx", index=False
        )

    return estimations
