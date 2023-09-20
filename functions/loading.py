import pandas as pd

from functions.merged_dataset_creation import create_preprocessed_dataset
from functions.preprocessing import outliers_preprocess


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
            path + "cgee_preprocessed_dataset_2023.parquet"
        )

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
            preprocessed_dataset.to_parquet(
                path + "cgee_preprocessed_dataset_2023.parquet"
            )

    if filter_outliers:
        for target in ["cf1", "cf2", "cf3", "cf123"]:
            preprocessed_dataset = outliers_preprocess(
                preprocessed_dataset,
                target,
                threshold_under=threshold_under,
                threshold_over=threshold_over,
            )

    return preprocessed_dataset
