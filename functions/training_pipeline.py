from functions.preprocessing import custom_train_split
from functions.results import best_model_analysis, metrics, results
import logging
import time
import configparser
import pandas as pd

config = configparser.ConfigParser()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def training_pipeline(
    targets,
    models,
    summary_final,
    summary_metrics_detailed,
    estimated_scopes,
    preprocessed_dataset,   
    restricted_features=False,
    save=False,
    ):
    """
    Apply a training pipeline for imputed targets, models, and parameters.

    This function performs a machine learning training pipeline on the provided data and settings.
    This function is a comprehensive training pipeline for machine learning models to predict target variables. It includes the following steps:

    1. Data Splitting: Splits the preprocessed dataset into training and testing sets for the given target variable.
    2. Model Training: Iterates over the provided machine learning models, trains each model, and evaluates its performance using RMSE (Root Mean Square Error).
    3. Model Selection: Selects the best-performing model based on the lowest RMSE.
    4. Feature Analysis: Conducts detailed feature analysis and visualization for the selected model.
    5. Results Saving: Optionally saves the results, trained models, and visualizations to specified directories if 'save' is set to True.

    Parameters:
    - path_Benchmark : The path to the benchmark dataset.
    - path_results : The path where the results will be saved.
    - path_models : The path where trained models will be saved.
    - path_intermediary : The path for storing intermediary data.
    - path_plot : The path for saving plots and visualizations.
    - targets : A list of target variables to predict.
    - models : A dictionary of machine learning models to use in the pipeline, where keys are model names
        and values are corresponding model classes or functions.
    - Summary_Final : A summary dictionary for storing final results.
    - ensemble : A list to store trained model instances.
    - summary_metrics_detailed : A summary dictionary for detailed metrics.
    - estimated_scopes : A dictionary to store estimated scopes.
    - preprocessed_dataset :The preprocessed dataset containing input features and target variables.
    - training_parameters :A dictionary containing training parameters such as 'extended_features', 'selec_sect', 'seed', 'n_iter', etc.
    - restricted_features : Whether to use restricted features during training (default is False).
    - save : Whether to save the results and trained models (default is False).

    Returns:
    - best_scores : A list of the best RMSE scores achieved for each target variable.
    - best_stds : A list of the standard deviations corresponding to the best RMSE scores.
    - summary_global : A summary dictionary for global metrics.
    - summary_metrics_detailed : Updated summary dictionary for detailed metrics after the training process.
    """
    best_scores = []
    best_stds = []
    coverage_train_dict = {}
    coverage_test_dict = {}
    difference_mean_train_dict = {}
    difference_mean_test_dict = {}
    coverage_lst = []


    if restricted_features:
        config.read('data/intermediary_data/restricted_features/parameters_restricted.ini')
        path = config["paths_restricted"]
        path_benchmark= path.get('path_benchmark')
        path_results= path.get('path_results')
        path_models=  path.get('path_models')
        path_intermediary=  path.get('path_intermediary')
        path_plot=  path.get('path_plot')
        path_rawdata=  path.get('path_rawdata')

    else :
        config.read('data/intermediary_data/unrestricted_features/parameters_unrestricted.ini')
        path = config["paths_unrestricted"]
        path_benchmark= path.get('path_benchmark')
        path_results= path.get('path_results')
        path_models=  path.get('path_models')
        path_intermediary=  path.get('path_intermediary')
        path_plot=  path.get('path_plot')
        path_rawdata=  path.get('path_rawdata')

    for target in targets:
        ensemble=[]
        parameters = config[target]
        training_parameters = {}
        training_parameters = {
                "seed": parameters.getint('seed'),
                "n_iter": parameters.getint('n_iter'),
                "extended_features": parameters.get('extended_features').split(','),
                "selec_sect": parameters.get('selec_sect').split(','),
                "cross_val": parameters.getboolean('cross_val')
            }            
        logger.info(f"Training for target: {target}")
        start_time = time.time()
        test_scores = []
        test_stds = []
        (X_train, y_train, X_test, y_test, df_test,df_train) = custom_train_split(
            preprocessed_dataset,
            path_benchmark,
            path_intermediary,
            target,
            extended_features=training_parameters["extended_features"],
            selec_sect=training_parameters["selec_sect"],
            restricted_features=restricted_features,
        )
        logger.info("Preprocessing done")
        seed = training_parameters["seed"]
        n_iter = training_parameters["n_iter"]
        for i, (model_name, model) in enumerate(models.items()):
            logger.info(f"Training model: {model_name}")
            model_i = model(
                X_train,
                y_train,
                cross_val=training_parameters["cross_val"],
                n_jobs=-1,
                verbose=0,
                n_iter=n_iter,
                seed=seed,

            )
            y_pred = model_i.predict(X_test)

            summary_global, rmse, std = metrics(
                y_test, y_pred, summary_final, target, model_name
            )
            ensemble.append(model_i)
            test_scores.append(rmse)
            test_stds.append(std)

            

        best_scores.append(test_scores[test_scores.index(min(test_scores))])
        best_stds.append(test_stds[test_scores.index(min(test_scores))])
        print(test_scores)
        logger.info("Modelisation done")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time for target {target}: {elapsed_time:.2f} seconds")

        if save:
            print(test_scores)
            best_model_index = test_scores.index(min(test_scores))
            best_model = ensemble[best_model_index]
            summary_metrics_detailed, estimated_scopes, lst = best_model_analysis(
                best_model,
                X_test,
                X_train,
                y_test,
                df_test,
                target,
                path_plot,
                preprocessed_dataset,
                path_intermediary,
                summary_metrics_detailed,
                estimated_scopes,
                restricted_features,
                path_models,
                path_rawdata,
            )


    if save:
        results(
            estimated_scopes, path_results, summary_metrics_detailed, summary_final, lst
        )

    return best_scores, best_stds, summary_global, summary_metrics_detailed
