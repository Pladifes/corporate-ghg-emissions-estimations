import warnings

import pandas as pd

from functions.loading import load_data
from functions.training_pipeline import training_pipeline
from functions.models import catboost_model

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def production_pipeline(
    restricted_features,
    path_rawdata="data/raw_data/",
    path_models="models/",
    path_benchmark="benchmark/",
    path_results="results/",
    path_plot="results/plot/",
    path_intermediary="data/intermediary_data/",
    targets=["cf1_log", "cf2_log", "cf3_log", "cf123_log"],
    models={"catboost": catboost_model},
):
    """
    Applies the full pipeline to train and save models.
    Models can thus be remployed
    """
    # Training parameters definition, has to be changed outside this function
    training_parameters = {
        "seed": 0,
        "n_iter": 10,
        "extended_features": [
            "revenue_log",
            "EMP_log",
            "asset_log",
            "NPPE_log",
            "CapEx_log",
            "Age",
            "CapInten",
            "GMAR",
            "Leverage",
            "Price",
            "fuel_intensity",
            "fiscal_year",
            "energy_consumed_log",
            "energy_produced_log",
            "INTAN_log",
            "AccuDep_log",
            "COGS_log",
        ],
        "selec_sect": ["gics_sub_ind", "gics_ind", "gics_group"],
        "cross_val": False,
    }
    summary_final = []
    ensemble = []
    summary_metrics_detailed = pd.DataFrame()
    estimated_scopes = []
    save = True

    # Dataset loading
    preprocessed_dataset = load_data(path_rawdata, save=save)

    # Preprocessing, training and outputs generation
    best_scores, best_stds, summary_global, summary_metrics_detailed = training_pipeline(
        path_benchmark=path_benchmark,
        path_results=path_results,
        path_models=path_models,
        path_intermediary=path_intermediary,
        path_plot=path_plot,
        targets=targets,
        models=models,
        summary_final=summary_final,
        ensemble=ensemble,
        summary_metrics_detailed=summary_metrics_detailed,
        estimated_scopes=estimated_scopes,
        preprocessed_dataset=preprocessed_dataset,
        training_parameters=training_parameters,
        restricted_features=restricted_features,
        save=save,
    )

    return
