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
    Merge multiple DataFrames containing diverse company-related data. The resulting DataFrame combines columns from each input DataFrame and introduces new columns.

    Parameters:
    - input_dataset (DataFrame): Financial data for companies.
    - carbon_pricing (DataFrame): Information on carbon pricing for countries and fiscal years.
    - income_group (DataFrame): Information on the income group of countries and fiscal years.
    - fuel_intensity (DataFrame): Information on the fuel intensity of countries and fiscal years.
    - region_mapping (DataFrame): Mapping of countries to regions.

    Returns: 
    - df_merged (DataFrame): The merged DataFrame containing data from all input DataFrames and newly created columns.
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

    Parameters:    
    - df: A DataFrame containing the necessary columns to calculate the financial ratios.
    
    Returns: 
    - The modified DataFrame with the calculated ratios.
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
    This function performs a series of data preprocessing tasks, including the addition of variables, data cleaning, and imputation. Some of the main steps include:
    - Adding variables to the DataFrame.
    - Removing rows with missing values in specific columns ('revenue' and 'gics_sub_ind').
    - Handling outliers in columns ('gmar', 'age', 'leverage', 'accu_dep').
    - Standardizing the 'fiscal_year' column data type to integer.
    - Imputing missing values in 'fuel_intensity' based on median values per fiscal year.
    - Assigning specific values to 'income_group' for 'Guernsey' and 'Jersey' in 'country_hq'.
    - Handling missing values in 'cf123' based on 'cf1', 'cf2', and 'cf3'.
    - Setting 'cf1' to the minimum positive value for rows where 'cf1' is zero.

    Parameters:
    - df (pandas.DataFrame): A pandas DataFrame containing the data to be preprocessed.

    Returns:
    - pandas.DataFrame: The preprocessed DataFrame.

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
    - 'input_dataset': The primary dataset containing financial and economic information.
    - 'carbon_pricing': Carbon pricing data after preprocessing.
    - 'income_group': Income group data after preprocessing.
    - 'fuel_intensity': Data related to fuel intensity.

    The function applies the following steps:
    1. Transforms and preprocesses 'carbon_pricing' data.
    2. Transforms and preprocesses 'income_group' data.
    3. Preprocesses 'CDP' data.
    4. Merges all datasets into 'df_merged'.
    5. Performs initial preprocessing on 'df_merged' using 'initial_preprocessing'.

    Parameters:
    - input_dataset (DataFrame): The primary dataset containing financial and economic information.
    - carbon_pricing (DataFrame): Carbon pricing data.
    - income_group (DataFrame): Income group data.
    - fuel_intensity (DataFrame): Data related to fuel intensity.
    - CDP (DataFrame): CDP dataset.

    Returns:
    - preprocessed_dataset (DataFrame): The preprocessed and merged dataset.
    """
    df_merged = merge_datasets(
        input_dataset, carbon_pricing, income_group, fuel_intensity, region_mapping
    )

    preprocessed_dataset = initial_preprocessing(df_merged)
    return preprocessed_dataset
