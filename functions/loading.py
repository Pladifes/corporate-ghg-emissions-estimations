import logging


import pandas as pd


from functions.merged_dataset_creation import create_preprocessed_dataset
from functions.preprocessing import outliers_preprocess


def country_region_mapping(path, df):
    """
    This function adds a "Region" columns to the Refinitiv data.
    """
    mapping = pd.read_excel(path + "country_region_mapping.xlsx")
    mapping_dict = mapping.set_index("Country").to_dict()["Region"]
    df["Region"] = df["CountryHQ"].apply(lambda x: mapping_dict[x])
    return df


def load_data(path, filter_outliers=True, threshold_under=1.5, threshold_over=2.5, save=False):
    """
    This function loads pre-downloaded datasets in the paths.
    Beware, if one is missing, code will return an error.
    """
    try:
        preprocessed_dataset = pd.read_parquet(path + "CGEE_preprocessed_dataset_2023.parquet")

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
        FuelIntensity = pd.read_csv(path + "2021FuelMix.csv", encoding="latin-1").rename(
            columns={"Value": "FuelIntensity"}
        )

        CDP = pd.read_excel(path + "CDP_filtered_for_CGEE_V1.xlsx")

        preprocessed_dataset = create_preprocessed_dataset(
            Refinitiv_data, CarbonPricing, IncomeGroup, FuelIntensity, CDP
        )
        if save:
            preprocessed_dataset.to_parquet(path + "CGEE_preprocessed_dataset_2023.parquet")

    preprocessed_dataset["CF1"] = preprocessed_dataset["CF1_merge"]
    preprocessed_dataset["CF2"] = preprocessed_dataset["CF2_merge"]
    preprocessed_dataset["CF3"] = preprocessed_dataset["CF3_merge"]
    preprocessed_dataset["CF123"] = preprocessed_dataset["CF123_merge"]
    preprocessed_dataset["CDP_CF2"] = preprocessed_dataset["CDP_CF2_location"]
    preprocessed_dataset["country_sector"] = (
        preprocessed_dataset["CountryHQ"].astype(str) + "_" + preprocessed_dataset["GICSSubInd"].astype(str)
    )

    if filter_outliers:
        for target in ["CF1_merge", "CF2_merge", "CF3_merge", "CF123_merge"]:
            preprocessed_dataset = outliers_preprocess(
                preprocessed_dataset, target, threshold_under=threshold_under, threshold_over=threshold_over
            )

    return preprocessed_dataset
