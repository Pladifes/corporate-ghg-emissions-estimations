import pandas as pd

from functions.merged_dataset_creation import create_preprocessed_dataset
from functions.preprocessing import outliers_preprocess

import configparser

config = configparser.ConfigParser()

def assert_input_format(df):
    """
    Asserts whether the input DataFrame contains a minimum set of required columns.

    This function checks whether the input DataFrame, 'df', contains a set of specific columns
    that are essential for further processing. If any of these columns are missing, it raises
    an AssertionError indicating that the input dataset does not meet the minimal required
    column criteria.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be checked for column presence.

    Raises:
    AssertionError: If the input DataFrame does not contain all the required columns.
    """

    mini = set(
        [
            "company_id",
            "company_name",
            "country_hq",
            "gics_name",
            "fiscal_year",
            "revenue",
            "ebit",
            "asset",
            "cf1",
            "cf2",
            "cf3",
            "cf123",
        ]
    )

    assert set(mini).intersection(set(df.columns)) == set(
        mini
    ), "input dataset does not contain minimal required columns"

    return


def load_data(
    filter_outliers=True, threshold_under=1.5, threshold_over=2.5, save=False
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

    """
    config.read('data/intermediary_data/restricted_features/parameters_restricted.ini')
    path_config = config["paths_restricted"]
    path=  path_config.get('path_rawdata')

    try:
        preprocessed_dataset = pd.read_parquet(
            path + "cgee_preprocessed_dataset_2023.parquet"
        )

    except FileNotFoundError:
        print("File not found, constructing it")

        input_dataset = pd.read_parquet(path + "input_dataset_full_extract.parquet")
        assert_input_format(input_dataset)
        region_mapping = pd.read_excel(path + "country_region_mapping.xlsx")
        carbon_pricing = pd.read_excel(
            path + "carbon_pricing_preprocessed_2023.xlsx",
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
