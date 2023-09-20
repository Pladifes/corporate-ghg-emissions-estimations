import numpy as np
import pandas as pd


def merge_datasets(
    input_dataset,
    carbon_pricing,
    income_group,
    fuel_intensity,
    region_mapping,
):
    """
    This function merges several DataFrames containing different types of data related to companies. The merged DataFrame includes columns from each input DataFrame as well as newly created columns.

    :param Refinitiv_data: A DataFrame containing financial data.
    :param carbon_pricing_transposed: A DataFrame containing information on carbon pricing for countries and fiscal years.
    :param income_group_transposed: A DataFrame containing information on the income group of countries and fiscal years.
    :param fuel_intensity_transposed: A DataFrame containing information on the fuel intensity of countries and fiscal years.
    :return: The merged DataFrame containing data from all input DataFrames and newly created columns.
    """
    df = pd.merge(
        input_dataset,
        carbon_pricing,
        how="left",
        left_on=["country_hq", "fiscal_year"],
        right_on=["tr_name", "fiscal_year"],
    )
    df = pd.merge(
        df,
        fuel_intensity[["area", "year", "fuel_intensity"]],
        how="left",
        left_on=["country_hq", "fiscal_year"],
        right_on=["area", "year"],
    )
    df_merged = pd.merge(
        df,
        income_group[["tr_name", "income_group", "fiscal_year"]],
        how="left",
        left_on=["country_hq", "fiscal_year"],
        right_on=["tr_name", "fiscal_year"],
    )
    mapping_dict = region_mapping.set_index("country").to_dict()["region"]
    df_merged["region"] = df_merged["country_hq"].apply(lambda x: mapping_dict[x])

    df_merged.drop(["tr_name_y", "tr_name_x"], axis=1, inplace=True)

    return df_merged


def add_variables(df):
    """
    This function takes a DataFrame 'df' and calculates three financial ratios using the columns in the DataFrame. The ratios are:

    Age: The age of a company, calculated as the ratio of Gross Property, Plant and Equipment (GPPE) to the difference between Earnings Before Interest, Taxes, Depreciation, and Amortization (ebitDA) and Earnings Before Interest and Taxes (ebit). If the denominator is 0, Age is set to NaN.
    CapInten: The capital intensity of a company, calculated as the ratio of Net Property, Plant, and Equipment (NPPE) to revenue. If the denominator is 0, CapInten is set to NaN.
    Leverage: The leverage of a company, calculated as the ratio of Long-Term Debt (LTDebt) to Total assets (asset). If the denominator is 0, Leverage is set to NaN.
    The function replaces any values of 'inf' in the calculated ratios with NaN. The modified DataFrame is then returned.

    :param df: A DataFrame containing the necessary columns to calculate the financial ratios.
    :return: The modified DataFrame with the calculated ratios.
    """
    df["age"] = df["gppe"] / (df["ebitda"] - df["ebit"])
    df["age"] = df["age"].replace(np.inf, np.nan)

    df["cap_inten"] = df["nppe"] / df["revenue"]
    df["cap_inten"] = df["cap_inten"].replace(np.inf, np.nan)

    df["leverage"] = df["lt_debt"] / df["asset"]
    df["leverage"] = df["leverage"].replace(np.inf, np.nan)
    return df


def initial_preprocessing(df):
    """
    A function to perform initial preprocessing on a pandas DataFrame.

    Args:
    df (pandas.DataFrame): A pandas DataFrame containing the data.
    target (str): The target variable to be used in the analysis.

    Returns:
    pandas.DataFrame: A preprocessed DataFrame.
    """
    df = add_variables(df)

    df.dropna(subset=["revenue", "gics_sub_ind"], inplace=True)

    df.loc[df[(df.gmar < 0) | (df.gmar > 100)].index, "gmar"] = np.nan
    df.loc[df[(df.age < 0) | (df.age > 100)].index, "age"] = np.nan
    df.loc[df[(df.leverage < 0) | (df.leverage > 1)].index, "leverage"] = np.nan
    df.loc[df[(df.accu_dep > 0)].index, "accu_dep"] = np.nan
    df["accu_dep"] = df.accu_dep * -1

    df["fiscal_year"] = df.fiscal_year.astype(int)

    medians_per_years = df.groupby("fiscal_year")["fuel_intensity"].median()
    for year in range(df.fiscal_year.min(), df.fiscal_year.max() + 1):
        df.loc[
            (df.fiscal_year == year) & (df.fuel_intensity.isna()), "fuel_intensity"
        ] = medians_per_years[year]

    df.loc[df[df.country_hq == "Guernsey"].index, "income_group"] = "H"
    df.loc[df[df.country_hq == "Jersey"].index, "income_group"] = "H"

    df.loc[df[(df.cf1.isna()) | (df.cf2.isna()) | (df.cf3.isna())].index, "cf123"] = [
        np.nan
        for i in range(len(df[(df.cf1.isna()) | (df.cf2.isna()) | (df.cf3.isna())]))
    ]

    df.loc[df[df.cf1 == 0].index, "cf1"] = [
        df[df.cf1 > 0].cf1.min() for i in range(len(df[df.cf1 == 0]))
    ]
    return df


def create_preprocessed_dataset(
    input_dataset, carbon_pricing, income_group, fuel_intensity, region_mapping
):
    """
    Creates a merged dataset from multiple data sources and performs preprocessing steps on it.

    This function integrates data from the following sources:
    - 'Refinitiv_data': The primary dataset containing financial and economic information.
    - 'CarbonPricing': Carbon pricing data after preprocessing.
    - 'IncomeGroup': Income group data after preprocessing.
    - 'FuelIntensity': Data related to fuel intensity.
    - 'CDP': CDP dataset after preprocessing.

    The function applies the following steps:
    1. Transforms and preprocesses 'CarbonPricing' data.
    2. Transforms and preprocesses 'IncomeGroup' data.
    3. Preprocesses 'CDP' data.
    4. Merges all datasets into 'df_merged'.
    5. Performs initial preprocessing on 'df_merged' using 'initial_preprocessing'.

    Parameters:
    - Refinitiv_data (DataFrame): The primary dataset containing financial and economic information.
    - CarbonPricing (DataFrame): Carbon pricing data.
    - IncomeGroup (DataFrame): Income group data.
    - FuelIntensity (DataFrame): Data related to fuel intensity.
    - CDP (DataFrame): CDP dataset.

    Returns:
    - preprocessed_dataset (DataFrame): The preprocessed and merged dataset.
    """
    df_merged = merge_datasets(
        input_dataset, carbon_pricing, income_group, fuel_intensity, region_mapping
    )

    preprocessed_dataset = initial_preprocessing(df_merged)
    return preprocessed_dataset
