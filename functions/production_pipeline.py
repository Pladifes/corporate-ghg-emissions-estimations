import warnings
import logging

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
    save
):
    """
    Apply a comprehensive pipeline for training machine learning models and saving them for future use.

    Parameters:
    - restricted_features (bool): Indicates whether to use restricted features during model training.
    - path_rawdata (str): Directory containing raw data.
    - path_models (str): Directory to save trained models.
    - path_benchmark (str): Directory for benchmark data.
    - path_results (str): Directory for storing results.
    - path_plot (str): Directory for saving plot images.
    - path_intermediary (str): Directory for intermediate data storage.
    - targets (list of str): List of target variables to train models for.
    - models (dict): Dictionary of model names and corresponding model functions.

    Returns:
    - None

    This function performs a complete machine learning pipeline, including data loading, preprocessing, outlier removal, model training, and result generation.

    The function adapts its behavior based on the 'restricted_features' flag. If it's set to True, it uses a different set of features and paths for data storage.

    The 'targets' parameter specifies the target variables to train models for, and 'models' is a dictionary containing model names as keys and their corresponding model functions.

    Training parameters are defined in the 'training_parameters' dictionary, including the random seed, the number of iterations, and the list of extended features to use for training. The 'cross_val' flag controls whether cross-validation is performed during training.

    The function saves the trained models and results in the specified directories. The summary results and estimated scopes are also generated and saved.
    """
    logging.info("Starting production pipeline")

    # Training parameters
 
    models = {
            "catboost": catboost_model,
    }

    targets = ["cf1", "cf2", "cf3", "cf123"]

    # Results containers
    summary_final=[]
    summary_metrics_detailed = pd.DataFrame()
    estimated_scopes = []

    preprocessed_dataset = load_data(save=save)
    best_scores, best_stds, summary_global, summary_metrics_detailed = training_pipeline(
    targets=targets,
    models=models,
    summary_final=summary_final,
    summary_metrics_detailed=summary_metrics_detailed,
    estimated_scopes=estimated_scopes,
    preprocessed_dataset=preprocessed_dataset,
    restricted_features=restricted_features,
    save=save,
  
)
    logging.info("Production pipeline completed")

    return
