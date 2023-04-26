import pickle as pkl
import pandas as pd
import numpy as np


from functions.merged_dataset_creation import CarbonPricing_preprocess, IncomeGroup_preprocess, merge_datasets
 
from functions.preprocessing import (
    encoding,
    set_columns
)

def apply_model_on_forbes_data(
    path_rawdata,    
    path_results,
    path_intermediary,
    path_models,
    GICSReclass,
    CarbonPricing,
    IncomeGroup,
    FuelIntensity,
    start_year=2008,
    end_year=2023,
    save=False
):
    """
    This function apply pre saved models to forbes data to predict scope 1, 2 and 3 emissions. 
    WARINING : models has to be restricted to Sales (Revenue), Profits (EBIT) and Assets (Asset), and GICSSubInd codes/names.
    """

    lst_df = []
    mapping = pd.read_excel(path_rawdata + "country_region_mapping.xlsx")
    mapping_dict = mapping.set_index("Country").to_dict()["Region"]

    for year in range(start_year, end_year):
        df_year = pd.read_excel(path_rawdata+"Forbes-all-with-sector.xlsx", sheet_name=str(year))
        df_year = df_year.drop_duplicates()
        df_year.dropna().reset_index(drop=True, inplace=True)

        df_year.replace(
            {'United States': "United States of America",
                'South Korea': "Korea; Republic (S. Korea)",
                'Ireland': "Ireland; Republic of",
                "Hong Kong/China": "China",
                "Australia/United Kingdom": "Australia",
                "Netherlands/United Kingdom": "Netherlands",
                "Panama/United Kingdom": "Panama",
                'Hong Kong-China': 'Hong Kong',
                'North America': "United States of America"}, inplace=True)

        df_year["Region"] = df_year["Country"].apply(lambda x: mapping_dict[x])

        del df_year["Market Value"]
        df_year["CF1"] = np.zeros(len(df_year))
        df_year["CF2"] = np.zeros(len(df_year))
        df_year["CF3"] = np.zeros(len(df_year))
        df_year["CF123"] = np.zeros(len(df_year))
        df_year = df_year.rename(columns={"Company Name": "Name", "Country": "CountryHQ", "Industry": "GICSName", "Sales": "Revenue", "Profits": "EBIT", "Assets": "Asset"})
        # df_year["FiscalYear"] = df_year.FiscalYear.astype(int) 

        CarbonPricing_Transposed = CarbonPricing_preprocess(CarbonPricing)
        IncomeGroup_Transposed = IncomeGroup_preprocess(IncomeGroup)


        raw_dataset_year = merge_datasets(
                            df_year,
                            GICSReclass,
                            CarbonPricing_Transposed,
                            IncomeGroup_Transposed,
                            FuelIntensity,
        )

        raw_dataset_year["FuelIntensity"] = raw_dataset_year.FuelIntensity.fillna(raw_dataset_year.FuelIntensity.median())

        raw_dataset_year = raw_dataset_year.rename(columns={"SubInd": "GICSSubInd"})
        raw_dataset_year = raw_dataset_year.drop_duplicates("Name")

        for scope in ["CF1", "CF2", "CF3", "CF123"]:
            dataset = encoding(raw_dataset_year, scope + "_log", path_intermediary, train=False, old_pipe=False, open_data=True, fill_grp="")
            features = pd.read_csv(path_intermediary + "features.csv").squeeze().tolist()
            dataset = set_columns(dataset, features)
            dataset = dataset[features]
            reg = pkl.load(open(path_models + "{}_log_model.pkl".format(scope), "rb"))
            scope_pred = reg.predict(dataset)
            df_year[scope + "_E"] = np.power(10, scope_pred + 1)

        df_year["CF1_E + CF2_E + CF3_E"] = df_year["CF1_E"] + df_year["CF2_E"] + df_year["CF3_E"] + df_year["CF123_E"]

        del df_year["CF1"]
        del df_year["CF2"]
        del df_year["CF3"]
        del df_year["CF123"]
        lst_df.append(df_year)

    df_all_years = pd.concat(lst_df).reset_index(drop=True)

    if save:
        df_all_years.to_excel(path_results + "Pladifes_free_emissions_estimates.xlsx", index=False)

    return df_all_years
