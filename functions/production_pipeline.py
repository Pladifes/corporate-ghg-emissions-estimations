import pandas as pd
import numpy as np

from functions.loading import load_data
from functions.merged_dataset_creation import create_preprocessed_dataset
from functions.training_pipeline import training_pipeline
from functions.models import xgboost_model, catboost_model, lgbm_model


def production_pipeline(targets, models, open_data,name_experiment):
    path_rawdata = "data/raw_data/"
    path_Benchmark = "benchmark/"
    Refinitiv_data, CarbonPricing, IncomeGroup, FuelIntensity, GICSReclass = load_data(
        path_rawdata
    )

    preprocessed_dataset = create_preprocessed_dataset(
        Refinitiv_data, GICSReclass, CarbonPricing, IncomeGroup, FuelIntensity
    )
    Summary_Final = []
    ensemble = []
    summary_metrics_detailed = pd.DataFrame()
    estimated_scopes = []

    if open_data:

        path_models = "models/open_data/"
        path_results = "results/open_data/"
        path_intermediary = "data/intermediary_data/open_data/"
        path_plot = "results/open_data/plot/"

        training_parameters = {
            "low": 0.01,
            "high": 1,
            "extended_features": ["Revenue_log", "Asset_log", "EBIT_log"],
            "selec_sect": ["GICSSubInd"],
            "fill_grp": "",
            "old_pipe": False,
            "cross_val": False,
        }
        (
            best_scores,
            best_stds,
            summary_global,
            summary_metrics_detailed,
        ) = training_pipeline(
            name_experiment=name_experiment,
            path_Benchmark=path_Benchmark,
            path_results=path_results,
            path_models=path_models,
            path_intermediary=path_intermediary,
            path_plot=path_plot,
            targets=targets,
            models=models,
            Summary_Final=Summary_Final,
            ensemble=ensemble,
            summary_metrics_detailed=summary_metrics_detailed,
            estimated_scopes=estimated_scopes,
            preprocessed_dataset=preprocessed_dataset,
            training_parameters=training_parameters,
            open_data=True,
            save=True,
        )

    else:
        path_models = "models/proprietary_data/"
        path_results = "results/proprietary_data/"
        path_plot = path_results + "plot/"
        path_intermediary = "data/intermediary_data/proprietary_data/"
        path_plot = "results/proprietary_data/plot/"

        training_parameters = {
            "low": 0.01,
            "high": 1,
            "extended_features": [
                "Revenue_log",
                "EMP_log",
                "Asset_log",
                "NPPE_log",
                "CapEx_log",
                "Age",
                "CapInten",
                "GMAR",
                "Leverage",
                "Price",
                "FuelIntensity",
                "FiscalYear",
                "ENEConsume_log",
                "ENEProduce_log",
                "INTAN_log",
                "AccuDep_log",
                "COGS_log",
            ],
            "selec_sect": ["GICSSubInd", "GICSInd", "GICSGroup"],
            "fill_grp": "",
            "old_pipe": False,
            "cross_val": False,
        }

        (
            best_scores,
            best_stds,
            summary_global,
            summary_metrics_detailed,
        ) = training_pipeline(
            name_experiment=name_experiment,
            path_Benchmark=path_Benchmark,
            path_results=path_results,
            path_models=path_models,
            path_intermediary=path_intermediary,
            path_plot=path_plot,
            targets=targets,
            models=models,
            Summary_Final=Summary_Final,
            ensemble=ensemble,
            summary_metrics_detailed=summary_metrics_detailed,
            estimated_scopes=estimated_scopes,
            preprocessed_dataset=preprocessed_dataset,
            training_parameters=training_parameters,
            open_data=False,
            save=True,
        )
    return
