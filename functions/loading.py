import logging


import pandas as pd


def country_region_mapping(path, df):
    mapping = pd.read_excel(path + "country_region_mapping.xlsx")
    mapping_dict = mapping.set_index("Country").to_dict()["Region"]
    df["Region"] = df["CountryHQ"].apply(lambda x: mapping_dict[x])
    return df


def load_data(path):
    """
    This function loads pre-downloaded datasets in the paths.
    Beware, if one is missing, code will return an error.
    """
    Refinitiv_data = pd.read_csv(
        path + "Refinitiv_cleaned_filtered.csv", encoding="latin-1", na_values=["."]
    )
    Refinitiv_data = country_region_mapping(path, Refinitiv_data)

    CarbonPricing = pd.read_excel(
        path + "Carbon Price Rework 20230405.xlsx",
    )
    IncomeGroup = pd.read_csv(
        path + "updated_income_group.csv",
        encoding="latin-1",
    )
    FuelIntensity = pd.read_csv(path + "2021FuelMix.csv", encoding="latin-1").rename(
        columns={"Value": "FuelIntensity"}
    )
    GICSReclass = pd.read_csv(
        path + "GICSSector.csv",
        encoding="latin-1",
    )

    return (
        Refinitiv_data,
        CarbonPricing,
        IncomeGroup,
        FuelIntensity,
        GICSReclass,
    )
