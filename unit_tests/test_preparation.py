import pandas as pd
import numpy as np
import pytest
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.data_preparation import load_data
from functions.data_preparation import CarbonPricing_preprocess
from functions.data_preparation import IncomeGroup_preprocess
from functions.data_preparation import merge_datasets
from functions.data_preparation import add_variables


path = "data/raw_data/"
Refinitiv_data, CarbonPricing, IncomeGroup, FuelIntensity, GICSReclass = load_data(path)
IncomeGroup_Transposed = IncomeGroup_preprocess(IncomeGroup, path)
CarbonPricing_Transposed = CarbonPricing_preprocess(CarbonPricing)
Refinitiv_data_test = Refinitiv_data.iloc[:5000,:]


def test_load_data():
    path = "data/raw_data/"
    Refinitiv_data, CarbonPricing, IncomeGroup, FuelIntensity, GICSReclass = load_data(path)
    assert Refinitiv_data.shape == (78073, 36)
    assert CarbonPricing.shape == (251, 84)
    assert IncomeGroup.shape == (215, 37)
    assert FuelIntensity.shape == (4670, 7)
    assert GICSReclass.shape == (165, 10)


def test_CarbonPricing_preprocess():
    expected_columns = set([
        "TR Name",
        "CO2Law",
        "CO2Scheme",
        "CO2Status",
        "CO2Coverage",
        "StartYear",
        "Price",
        "FiscalYear",
        "Status",
    ])
    assert set(CarbonPricing_Transposed.columns.tolist()) == expected_columns


def test_IncomeGroup_preprocess():
    assert IncomeGroup_Transposed.columns.tolist() == ["TR Name", "Region", "FiscalYear", "IncomeGroup"]
    assert IncomeGroup_Transposed["IncomeGroup"].dtype == object
    assert len(IncomeGroup_Transposed) == 3462


def test_merge_datasets():
    
    result = merge_datasets(
        Refinitiv_data_test,
        GICSReclass,
        CarbonPricing_Transposed,
        IncomeGroup_Transposed,
        FuelIntensity,
    )

    assert result["FuelIntensity"].dtype == float
    assert result["Price"].dtype == float
    assert result["FiscalYear"].dtype == 'int64'
    assert len(result) == len(Refinitiv_data_test)


def test_add_variables():
    df = pd.DataFrame(
        {
            "GPPE": [100, 200, 300],
            "EBITDA": [50, 75, 100],
            "EBIT": [20, 30, 40],
            "NPPE": [50, 75, 100],
            "Revenue": [500, 750, 1000],
            "LTDebt": [25, 50, 75],
            "Asset": [200, 300, 400],
        }
    )
    result = add_variables(df)
    assert result["Age"][1] == 4.444444444444445 
    assert result["CapInten"][0] == 0.1
    assert result["Leverage"][0] == 0.125

