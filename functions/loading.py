import logging

import pandas as pd

from functions.merged_dataset_creation import create_preprocessed_dataset
from functions.preprocessing import outliers_preprocess


def country_region_mapping(path, df):
    """
    Add a "Region" column to a DataFrame using a country-to-region mapping.

    Parameters:
    - path (str): The path to the directory containing the country-to-region mapping file.
    - df (DataFrame): The DataFrame to which the "Region" column will be added.

    Returns:
    - df (DataFrame): The input DataFrame with the "Region" column added.

    This function reads a country-to-region mapping file and adds a "Region" column to the input DataFrame 'df' based on the mapping.
    The mapping file should contain two columns: "Country" and "Region". The function uses the "CountryHQ" column from the input DataFrame to look up the corresponding region in the mapping and adds it as a new "Region" column in the DataFrame.

    Example usage:
    mapped_df = country_region_mapping("data/", raw_data)
    """
    mapping = pd.read_excel(path + "country_region_mapping.xlsx")
    mapping_dict = mapping.set_index("Country").to_dict()["Region"]
    df["Region"] = df["CountryHQ"].apply(lambda x: mapping_dict[x])
    return df


def load_data(
    path, filter_outliers=True, threshold_under=1.5, threshold_over=2.5, save=False
):
    """
    Load pre-downloaded datasets from the specified path or construct them if missing.

    Parameters:
    - path (str): The base path where datasets are stored or will be saved.
    - filter_outliers (bool): Whether to filter outliers in the loaded dataset.
    - threshold_under (float): The lower threshold for outlier removal.
    - threshold_over (float): The upper threshold for outlier removal.
    - save (bool): Whether to save the constructed dataset if it needs to be created.

    Returns:
    - preprocessed_dataset (DataFrame): The loaded or constructed preprocessed dataset.

    This function attempts to load a pre-downloaded preprocessed dataset from the specified path. If the dataset is not found, it constructs it from other source files and optionally saves it. The construction involves processing Refinitiv data, CarbonPricing data, IncomeGroup data, FuelIntensity data, and CDP data to create a preprocessed dataset.

    After loading or constructing the dataset, it renames columns and creates additional columns for CF1, CF2, CF3, CF123, CDP_CF2, and country_sector.

    If 'filter_outliers' is set to True, the function performs outlier removal on specific target columns using the provided thresholds.

    Example usage:
    data = load_data("data/", filter_outliers=True, threshold_under=1.5, threshold_over=2.5, save=True)
    """
    try:
        preprocessed_dataset = pd.read_parquet(
            path + "CGEE_preprocessed_dataset_2023.parquet"
        )

    except FileNotFoundError:
        print("File not found, constructing it")
        Refinitiv_data = pd.read_parquet(path + "refinitiv_cleaned_2023.parquet")
        Refinitiv_data = country_region_mapping(path, Refinitiv_data)

        CarbonPricing = pd.read_excel(
            path + "Carbon Price Rework 20230405.xlsx",
        )
        IncomeGroup = pd.read_excel(
            path + "updated_income_group.xlsx",
        )
        FuelIntensity = pd.read_csv(
            path + "2021FuelMix.csv", encoding="latin-1"
        ).rename(columns={"Value": "FuelIntensity"})

        CDP = pd.read_excel(path + "CDP_filtered_for_CGEE_V1.xlsx")

        preprocessed_dataset = create_preprocessed_dataset(
            Refinitiv_data, CarbonPricing, IncomeGroup, FuelIntensity, CDP
        )
        if save:
            preprocessed_dataset.to_parquet(
                path + "CGEE_preprocessed_dataset_2023.parquet"
            )

    preprocessed_dataset["CF1"] = preprocessed_dataset["CF1_merge"]
    preprocessed_dataset["CF2"] = preprocessed_dataset["CF2_merge"]
    preprocessed_dataset["CF3"] = preprocessed_dataset["CF3_merge"]
    preprocessed_dataset["CF123"] = preprocessed_dataset["CF123_merge"]
    preprocessed_dataset["CDP_CF2"] = preprocessed_dataset["CDP_CF2_location"]
    preprocessed_dataset["country_sector"] = (
        preprocessed_dataset["CountryHQ"].astype(str)
        + "_"
        + preprocessed_dataset["GICSSubInd"].astype(str)
    )

    if filter_outliers:
        for target in ["CF1_merge", "CF2_merge", "CF3_merge", "CF123_merge"]:
            preprocessed_dataset = outliers_preprocess(
                preprocessed_dataset,
                target,
                threshold_under=threshold_under,
                threshold_over=threshold_over,
            )

    return preprocessed_dataset
