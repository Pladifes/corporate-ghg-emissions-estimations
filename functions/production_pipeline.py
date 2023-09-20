import warnings

import pandas as pd

from functions.loading import load_data
from functions.training_pipeline import training_pipeline
from functions.models import catboost_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Apply the full pipeline to train and save models, allowing for reuse of models.

    Parameters:
    - restricted_features (list): A list of features to be restricted or excluded during model training.

    This function applies a comprehensive pipeline for training machine learning models and saving them for future use.
    The pipeline includes data loading, preprocessing, outlier removal, model training, and output generation.

    Parameters:
    - restricted_features (list): A list of feature names to be restricted or excluded during model training.

    The function uses the following directories and settings for its operation:
    - path_rawdata: Directory containing raw data.
    - path_models: Directory where trained models will be saved.
    - path_Benchmark: Directory for benchmark data.
    - path_results: Directory for storing results.
    - path_plot: Directory for saving plot images.
    - path_intermediary: Directory for intermediate data storage.

    It trains models for the specified target variables and uses a dictionary of models, where the keys are model names
    and the values are the corresponding model functions.

    The 'training_parameters' dictionary includes various parameters for training, such as the random seed, the number of iterations,
    and the list of extended features to use.

    The function returns no explicit output but performs the entire pipeline, including loading the dataset, preprocessing,
    removing outliers, training models, and generating results. The trained models and results are saved in the specified directories.

    Note: This function is designed for a specific use case and may require additional context to understand its full functionality.
    """
    logging.info("Starting production pipeline")
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

    preprocessed_dataset = load_data(path_rawdata, save=save)

    (
        best_scores,
        best_stds,
        summary_global,
        summary_metrics_detailed,
    ) = training_pipeline(
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
    logging.info("Production pipeline completed")

    return
