import pandas as pd
import numpy as np
import pytest
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.data_preparation import merge_datasets
from functions.data_preparation import add_variables
from functions.data_preparation import load_data
from functions.data_preparation import CarbonPricing_preprocess
from functions.data_preparation import IncomeGroup_preprocess

from functions.preprocessing import GICS_to_name
from functions.preprocessing import special_case
from functions.preprocessing import country_region_mapping_incomegroup
from functions.preprocessing import initial_preprocessing
from functions.preprocessing import df_split
from functions.preprocessing import set_columns
from functions.preprocessing import label_FinalCO2Law
from functions.preprocessing import processingrawdata
from functions.preprocessing import selected_features
from functions.preprocessing import outliers_preprocess
from functions.preprocessing import set_columns

path = "data/raw_data/"
Refinitiv_data, CarbonPricing, IncomeGroup, FuelIntensity, GICSReclass = load_data(path)
path_Benchmark = "C:/Users/mohamed.fahmaoui/Projets/scopes_estimations/corporate_ghg_estimation/Benchmark/"
CarbonPricing_Transposed = CarbonPricing_preprocess(CarbonPricing)
IncomeGroup_Transposed = IncomeGroup_preprocess(IncomeGroup, path)
df_merged = merge_datasets(
    Refinitiv_data,
    GICSReclass,
    CarbonPricing_Transposed,
    IncomeGroup_Transposed,
    FuelIntensity,
)
raw_dataset = add_variables(df_merged)


def test_GICS_to_name():
    assert GICS_to_name(10.0) == "Energy"
    assert GICS_to_name(15.0) == "Materials"
    assert GICS_to_name(20.0) == "Industrials"
    assert GICS_to_name(25.0) == "Cons. Discretionary"
    assert GICS_to_name(30.0) == "Cons. Staples"
    assert GICS_to_name(35.0) == "Health Care"
    assert GICS_to_name(40.0) == "Financials"
    assert GICS_to_name(45.0) == "IT"
    assert GICS_to_name(50.0) == "Telecommunication"
    assert GICS_to_name(55.0) == "Utilities"
    assert GICS_to_name(60.0) == "Real Estate"
    assert GICS_to_name(65.0) == 65.0


def test_special_case():
    data = {
        "FinalEikonID": ["SSABa.ST", "OMH.AX", "NHH.MC", "1101.HK"],
        "FiscalYear": [2005, 2006, 2006, 2006],
    }
    df = pd.DataFrame(data)

    df_out = special_case(df)
    assert len(df_out) == 0
    assert "SSABa.ST" not in df_out["FinalEikonID"].values
    assert "OMH.AX" not in df_out["FinalEikonID"].values
    assert "NHH.MC" not in df_out["FinalEikonID"].values
    assert "1101.HK" not in df_out["FinalEikonID"].values

    data = {
        "FinalEikonID": ["AAPL.O", "MSFT.O"],
        "FiscalYear": [2005, 2006],
    }
    df = pd.DataFrame(data)
    df_out = special_case(df)
    assert len(df_out) == 2
    assert "AAPL.O" in df_out["FinalEikonID"].values
    assert "MSFT.O" in df_out["FinalEikonID"].values


def test_country_region_mapping_incomegroup():
    df = pd.DataFrame({"TR Name": ["Canada", "Mexico", "United States"]})
    mapping_data = {
        "Country": ["Canada", "Mexico", "United States"],
        "Region": ["North America", "North America", "North America"],
    }
    mapping = pd.DataFrame(mapping_data)
    mapping_file = "country_region_mapping.xlsx"
    mapping.to_excel(mapping_file, index=False)
    result_df = country_region_mapping_incomegroup("", df)
    assert result_df.iloc[0]["Region"] == "North America"
    assert result_df.iloc[1]["Region"] == "North America"
    assert result_df.iloc[2]["Region"] == "North America"
    os.remove(mapping_file)


def test_initial_preprocessing():

    raw_data_test = raw_dataset.iloc[:5000, :]
    processed_data = initial_preprocessing(raw_data_test, "CF1_log", path)
    assert "SectorName" in processed_data.columns
    assert "NaN" not in processed_data["SectorName"].unique()


def test_df_split():
    df_train, df_test = df_split(raw_dataset, path_Benchmark)
    assert df_train["Name"].unique() != df_test["Name"].unique()


def test_set_columns():
    df = pd.DataFrame({"col_1": [1, 2, 3]})
    features = ["col_1", "col_3"]
    result = set_columns(df, features)
    expected_columns = ["col_1", "col_3"]
    assert list(result.columns) == expected_columns

    expected_df = pd.DataFrame(
        {"col_1": [1, 2, 3], "col_3": [0, 0, 0]}
    )
    assert result.equals(expected_df)

def test_label_FinalCO2Law():
    input_row = pd.Series({
        "CO2Law": "Yes",
        "CO2Status": "Implemented",
        "CO2Coverage": "Subnational"
    })
    expected_output = "Subnational Implemented"
    assert label_FinalCO2Law(input_row) == expected_output

def test_processingrawdata():
    data_old = pd.DataFrame({
        'CO2Law': ['Yes', 'No', 'Yes', 'TBD'],
        'CO2Status': ['Scheduled', 'Implemented', 'Implemented', 'Scheduled'],
        'CO2Coverage': ['Subnational', 'National', 'Regional', 'National'],
        'GICSGroup': ['Group A', 'Group B', 'Group A', 'Group B'],
        'GICSSector': ['Sector A', 'Sector B', 'Sector A', 'Sector B'],
        'Recat': ['SubInd A', 'SubInd B', 'SubInd A', 'SubInd B'],
        'Recat2': ['Ind A', 'Ind B', 'Ind A', 'Ind B'],
        'IncomeGroup': ['High', 'Low', 'Low', 'Middle'],
        'FiscalYear': [2018, 2019, 2019, 2020],
        'GICSSubInd': ['SubInd A', 'SubInd B', 'SubInd A', 'SubInd B'],
        'GICSInd': ['Ind A', 'Ind B', 'Ind A', 'Ind B'],
        'GMAR': [10, 20, 30, None]
    })
    
    data_new = processingrawdata(data_old, fillGMAR=True)
    
    assert data_new.shape == (4, 31)
    assert 'FinalCO2Law__No CO2 Law' in data_new.columns
    assert 'GICSSector__Sector A' in data_new.columns
    assert 'IncomeGroup__Middle' in data_new.columns
    assert data_new['GMAR'].isna().sum() == 0

def test_selected_features():
   features = selected_features(raw_dataset,'CF2_log',path)
    assert len(features) ==302

def test_outliers_preprocess():
    df = pd.DataFrame({
        "CF1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "CF2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "CF3": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "CF123": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    })

    data_new = outliers_preprocess(df)
    expected_df = pd.DataFrame({
        "CF1": [2, 3, 4, 5, 6, 7, 8, 9],
        "CF2": [12, 13, 14, 15, 16, 17, 18, 19],
        "CF3": [22, 23, 24, 25, 26, 27, 28, 29],
        "CF123": [32, 33, 34, 35, 36, 37, 38, 39]
    })
    assert data_new.shape == expected_df.shape
