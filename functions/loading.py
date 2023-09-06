import logging


import pandas as pd


from functions.merged_dataset_creation import create_preprocessed_dataset
from functions.preprocessing import outliers_preprocess


def load_data(path, filter_outliers=True, threshold_under=1.5, threshold_over=2.5, save=False):
    """
    This function loads pre-downloaded datasets in the paths.
    Beware, if one is missing, code will return an error.
    """
    try:
        preprocessed_dataset = pd.read_parquet(path + "cgee_preprocessed_dataset_2023.parquet")

    except FileNotFoundError:
        print("File not found, constructing it")

        input_dataset = pd.read_parquet(path + "input_dataset.parquet")
        region_mapping = pd.read_excel(path + "country_region_mapping.xlsx")
        carbon_pricing = pd.read_csv(
            path + "carbon_pricing_preprocessed_2023.csv",
        )
        income_group = pd.read_csv(
            path + "income_group_preprocessed_2023.csv",
        )
        fuel_intensity = pd.read_csv(path + "fuel_mix_2023.csv")

        preprocessed_dataset = create_preprocessed_dataset(
            input_dataset, carbon_pricing, income_group, fuel_intensity, region_mapping
        )
        if save:
            preprocessed_dataset.to_parquet(path + "cgee_preprocessed_dataset_2023.parquet")

    if filter_outliers:
        for target in ["cf1", "cf2", "cf3", "cf123"]:
            preprocessed_dataset = outliers_preprocess(
                preprocessed_dataset, target, threshold_under=threshold_under, threshold_over=threshold_over
            )

    return preprocessed_dataset
