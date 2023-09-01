import warnings

import pandas as pd

from functions.loading import load_data
from functions.preprocessing import outliers_preprocess
from functions.training_pipeline import training_pipeline
from functions.models import xgboost_model, catboost_model, lgbm_model

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def production_pipeline(restricted_features):
    """
    Applies the full pipeline to train and save models.
    Models can thus be remployed
    """
    # Parameters definition
    path_rawdata = "data/raw_data/"
    path_models = "models/"
    path_Benchmark = "Benchmark/"
    path_results = "results/"
    path_plot = path_results + "plot/"
    path_intermediary = "data/intermediary_data/"

    targets = ["CF1_log", "CF2_log", "CF3_log", "CF123_log"]
    models = {
        # "xgboost": xgboost_model,
        "catboost": catboost_model,
        # "lgbm": lgbm_model,
    }
    training_parameters = {
        "seed": 0,
        "n_iter": 10,
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
        "cross_val": False,
    }
    Summary_Final = []
    ensemble = []
    summary_metrics_detailed = pd.DataFrame()
    estimated_scopes = []
    save = True
    models = {
        # "xgboost": xgboost_model,
        "catboost": catboost_model,
        # "lgbm": lgbm_model,
    }

    # Dataset loading
    preprocessed_dataset = load_data(path_rawdata, save=save)

    # Datasset preprocessing
    preprocessed_dataset["CF1"] = preprocessed_dataset["CF1_merge"]
    preprocessed_dataset["CF2"] = preprocessed_dataset["CF2_merge"]
    preprocessed_dataset["CF3"] = preprocessed_dataset["CF3_merge"]
    preprocessed_dataset["CF123"] = preprocessed_dataset["CF123_merge"]
    preprocessed_dataset["CDP_CF2"] = preprocessed_dataset["CDP_CF2_location"]

    # Global outlier removal
    threshold_under = 1.5
    threshold_over = 2.5
    for target in ["CF1_merge", "CF2_merge", "CF3_merge", "CF123_merge"]:
        preprocessed_dataset = outliers_preprocess(
            preprocessed_dataset, target, threshold_under=threshold_under, threshold_over=threshold_over
        )

    # Preprocessing, training and outputs generation
    best_scores, best_stds, summary_global, summary_metrics_detailed = training_pipeline(
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
        restricted_features=restricted_features,
        save=save,
    )

    return
