import logging


import numpy as np
import pandas as pd


def CarbonPricing_preprocess(CarbonPricing):
    """
    This code snippet performs the following operations:
    1. Selects specific columns from a pandas DataFrame called CarbonPricing.
    2. Transposes the selected columns into rows.
    3. Replaces any instances of "No" or "TBD" in the "Status" column of the transposed DataFrame with "No" and sets the corresponding values in the "CO2Law", "CO2Scheme", "CO2Status", and "CO2Coverage" columns to None.

    """
    CarbonPricing = CarbonPricing[
        [
            "TR Name",
            "CO2Law",
            "CO2Scheme",
            "CO2Status",
            "CO2Coverage",
            "StartYear",
            "2004",
            "2005",
            "2006",
            "2007",
            "2008",
            "2009",
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "price_2004",
            "price_2005",
            "price_2006",
            "price_2007",
            "price_2008",
            "price_2009",
            "price_2010",
            "price_2011",
            "price_2012",
            "price_2013",
            "price_2014",
            "price_2015",
            "price_2016",
            "price_2017",
            "price_2018",
            "price_2019",
            "price_2020",
            "price_2021",
        ]
    ]

    price_cols = CarbonPricing.filter(regex="^price").columns.tolist()
    status_cols = CarbonPricing.filter(regex="^20").columns.tolist()

    CarbonPricing_Transposed1 = pd.melt(
        CarbonPricing,
        id_vars=[
            "TR Name",
            "CO2Law",
            "CO2Scheme",
            "CO2Status",
            "CO2Coverage",
            "StartYear",
        ],
        value_vars=price_cols,
        var_name="Year",
        value_name="Price",
    )

    CarbonPricing_Transposed1["FiscalYear"] = (
        CarbonPricing_Transposed1["Year"].str.extract(r"price_(\d{4})").astype(int)
    )
    CarbonPricing_Transposed1 = CarbonPricing_Transposed1.drop(columns=["Year"])
    CarbonPricing_Transposed1 = CarbonPricing_Transposed1.sort_values(
        by=["TR Name", "FiscalYear"]
    ).reset_index(drop=True)

    CarbonPricing_Transposed = pd.melt(
        CarbonPricing,
        id_vars=[
            "TR Name",
            "CO2Law",
            "CO2Scheme",
            "CO2Status",
            "CO2Coverage",
            "StartYear",
        ],
        value_vars=status_cols,
        var_name="FiscalYear",
        value_name="Status",
    )
    CarbonPricing_Transposed = CarbonPricing_Transposed.sort_values(
        by=["TR Name", "FiscalYear"]
    ).reset_index(drop=True)
    CarbonPricing_Transposed["Price"] = CarbonPricing_Transposed1["Price"]

    dict_names = {
        "Korea (South)": "Korea; Republic (S. Korea)",
        "USA": "United States of America",
        "British Virgin Islands": "Virgin Islands; British",
    }
    CarbonPricing_Transposed["TR Name"] = CarbonPricing_Transposed["TR Name"].replace(
        dict_names
    )

    mask = CarbonPricing_Transposed["Status"].isin(["No", "TBD"])
    CarbonPricing_Transposed.loc[
        mask, ["CO2Law", "CO2Scheme", "CO2Status", "CO2Coverage"]
    ] = ("No", None, None, None)
    CarbonPricing_Transposed["FiscalYear"] = CarbonPricing_Transposed.FiscalYear.astype(int)

    return CarbonPricing_Transposed


def IncomeGroup_preprocess(IncomeGroup):
    """
    This code transposes the input dataframe 'IncomeGroup' by setting the index to the columns
    ["DS Country", "ISO Country", "TR Name", "Region"], and stacking all other columns to create a single
    'IncomeGroup' column. It then resets the index, and renames the stacked column to 'IncomeGroup' and the
    original stacked index to 'FiscalYear'. The resulting dataframe is assigned to 'IncomeGroup_Transposed'.

    Returns:
    --------
    IncomeGroup_Transposed : pandas.DataFrame
        A transposed version of the 'IncomeGroup' dataframe, with the columns "DS Country", "ISO Country",
        "TR Name", "Region", "FiscalYear", and "IncomeGroup".
    """
    IncomeGroup_Transposed = (
        IncomeGroup.set_index(["TR Name"])
        .stack()
        .reset_index()
    )
    IncomeGroup_Transposed = IncomeGroup_Transposed[IncomeGroup_Transposed.level_1 != "code"]
    IncomeGroup_Transposed = IncomeGroup_Transposed.rename(columns={0: "IncomeGroup", "level_1": "FiscalYear"})

    IncomeGroup_Transposed["IncomeGroup"] = IncomeGroup_Transposed[
        "IncomeGroup"
    ].str.replace(";", "")
    IncomeGroup_Transposed["FiscalYear"] = (
        IncomeGroup_Transposed["FiscalYear"].str.replace(";", "").astype(int)
    )
    IncomeGroup_Transposed = IncomeGroup_Transposed[
        IncomeGroup_Transposed["FiscalYear"] >= 2005
    ] 

    return IncomeGroup_Transposed


def FuelIntensity_preprocess(FuelIntensity):
    """
    This code transposes the input dataframe 'FuelIntensity' by setting the index to the columns
    ["DS", "ISO", "TR", "IEAName"], and stacking all other columns to create a single 'FuelIntensity'
    column. It then resets the index, and renames the stacked column to 'FuelIntensity' and the original
    stacked index to 'FiscalYear'. The resulting dataframe is assigned to 'FuelIntensity_Transposed'.

    Returns:
    --------
    FuelIntensity_Transposed : pandas.DataFrame
        A transposed version of the 'FuelIntensity' dataframe, with the columns "DS", "ISO", "TR",
        "IEAName", "FiscalYear", and "FuelIntensity".
    """

    FuelIntensity_Transposed = (
        FuelIntensity.set_index(["DS", "ISO", "TR", "IEAName"])
        .stack()
        .reset_index()
        .rename(columns={0: "FuelIntensity", "level_4": "FiscalYear"})
    )

    FuelIntensity_Transposed["FuelIntensity"] = FuelIntensity_Transposed.FuelIntensity.astype(float)
    FuelIntensity_Transposed["FiscalYear"] = FuelIntensity_Transposed.FiscalYear.astype(int)

    FuelIntensity_Transposed.rename(columns={"TR":"Area", "FiscalYear":"Year"})

    logging.info("FuelIntensity database preprocessed")

    return FuelIntensity_Transposed


def merge_datasets(
    Refinitiv_data,
    GICSReclass,
    CarbonPricing_Transposed,
    IncomeGroup_Transposed,
    FuelIntensity_Transposed,
):
    """
    This function merges several DataFrames containing different types of data related to companies. The merged DataFrame includes columns from each input DataFrame as well as newly created columns.

    :param Refinitiv_data: A DataFrame containing financial data.
    :param GICSReclass: A DataFrame containing a reclassified version of the Global Industry Classification Standard (GICS) Sub-Industry (SubInd) column.
    :param CarbonPricing_Transposed: A DataFrame containing information on carbon pricing for countries and fiscal years.
    :param IncomeGroup_Transposed: A DataFrame containing information on the income group of countries and fiscal years.
    :param FuelIntensity_Transposed: A DataFrame containing information on the fuel intensity of countries and fiscal years.
    :return: The merged DataFrame containing data from all input DataFrames and newly created columns.
    """
    df = pd.merge(
        Refinitiv_data,
        CarbonPricing_Transposed,
        how="left",
        left_on=["CountryHQ", "FiscalYear"],
        right_on=["TR Name", "FiscalYear"],
    )
    df = pd.merge(
        df,
        FuelIntensity_Transposed[["Area", "Year", "FuelIntensity"]],
        how="left",
        left_on=["CountryHQ", "FiscalYear"],
        right_on=["Area", "Year"],
    )
    df = pd.merge(
        df,
        IncomeGroup_Transposed[["TR Name", "IncomeGroup", "FiscalYear"]],
        how="left",
        left_on=["CountryHQ", "FiscalYear"],
        right_on=["TR Name", "FiscalYear"],
    )

    df_merged = pd.merge(
        df,
        GICSReclass[["SubInd", "SubIndName", "Recat", "Recat2"]],
        how="left",
        left_on=["GICSName"],
        right_on=["SubIndName"],
    )

    df_merged.drop(["TR Name_y", "TR Name_x"], axis=1, inplace=True)

    return df_merged


def add_variables(df):
    """
    This function takes a DataFrame 'df' and calculates three financial ratios using the columns in the DataFrame. The ratios are:

    Age: The age of a company, calculated as the ratio of Gross Property, Plant and Equipment (GPPE) to the difference between Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA) and Earnings Before Interest and Taxes (EBIT). If the denominator is 0, Age is set to NaN.
    CapInten: The capital intensity of a company, calculated as the ratio of Net Property, Plant, and Equipment (NPPE) to Revenue. If the denominator is 0, CapInten is set to NaN.
    Leverage: The leverage of a company, calculated as the ratio of Long-Term Debt (LTDebt) to Total Assets (Asset). If the denominator is 0, Leverage is set to NaN.
    The function replaces any values of 'inf' in the calculated ratios with NaN. The modified DataFrame is then returned.

    :param df: A DataFrame containing the necessary columns to calculate the financial ratios.
    :return: The modified DataFrame with the calculated ratios.
    """
    df["Age"] = df["GPPE"] / (df["EBITDA"] - df["EBIT"])
    df["Age"] = df["Age"].replace(np.inf, np.nan)

    df["CapInten"] = df["NPPE"] / df["Revenue"]
    df["CapInten"] = df["CapInten"].replace(np.inf, np.nan)

    df["Leverage"] = df["LTDebt"] / df["Asset"]
    df["Leverage"] = df["Leverage"].replace(np.inf, np.nan)
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

    df.dropna(subset=["Revenue", "GICSSector"], inplace=True)

    df = df.drop(df[(df.GMAR < 0) | (df.GMAR > 100)].index)
    df = df.drop(df[(df.Age < 0) | (df.Age > 100)].index)
    df = df.drop(df[(df.Leverage < 0) | (df.Leverage > 1)].index)
    df = df.drop(df[(df.AccuDep > 0)].index)
    df["AccuDep"] = df.AccuDep * -1

    medians_per_years = df.groupby("FiscalYear")["FuelIntensity"].median()
    for year in range(2002, 2022):
        df.loc[
            (df.FiscalYear == year) & (df.FuelIntensity.isna()), "FuelIntensity"
        ] = medians_per_years[year]

    return df


def create_preprocessed_dataset(
    Refinitiv_data,
    GICSReclass,
    CarbonPricing,
    IncomeGroup,
    FuelIntensity,
):
    """
    Create the merged dataset and preprocess it with target independant steps.
    """
    CarbonPricing_Transposed = CarbonPricing_preprocess(CarbonPricing)
    IncomeGroup_Transposed = IncomeGroup_preprocess(IncomeGroup)
    df_merged = merge_datasets(
                            Refinitiv_data,
                            GICSReclass,
                            CarbonPricing_Transposed,
                            IncomeGroup_Transposed,
                            FuelIntensity,
    )

    preprocessed_dataset = initial_preprocessing(df_merged)
    return preprocessed_dataset
