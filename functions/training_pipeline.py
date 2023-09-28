from functions.preprocessing import custom_train_split
from functions.results import best_model_analysis, metrics, results
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def training_pipeline(
    path_benchmark,
    path_results,
    path_models,
    path_intermediary,
    path_plot,
    targets,
    models,
    summary_final,
    ensemble,
    summary_metrics_detailed,
    estimated_scopes,
    preprocessed_dataset,
    training_parameters,
    restricted_features=False,
    save=False,
):
    """
    Apply a training pipeline for imputed targets, models, and parameters.

    This function performs a machine learning training pipeline on the provided data and settings.

    Parameters:
    -----------
    path_Benchmark : str
        The path to the benchmark dataset.

    path_results : str
        The path where the results will be saved.

    path_models : str
        The path where trained models will be saved.

    path_intermediary : str
        The path for storing intermediary data.

    path_plot : str
        The path for saving plots and visualizations.

    targets : list
        A list of target variables to predict.

    models : dict
        A dictionary of machine learning models to use in the pipeline, where keys are model names
        and values are corresponding model classes or functions.

    Summary_Final : dict
        A summary dictionary for storing final results.

    ensemble : list
        A list to store trained model instances.

    summary_metrics_detailed : dict
        A summary dictionary for detailed metrics.

    estimated_scopes : dict
        A dictionary to store estimated scopes.

    preprocessed_dataset : pd.DataFrame
        The preprocessed dataset containing input features and target variables.

    training_parameters : dict
        A dictionary containing training parameters such as 'extended_features', 'selec_sect', 'seed', 'n_iter', etc.

    restricted_features : bool, optional
        Whether to use restricted features during training (default is False).

    save : bool, optional
        Whether to save the results and trained models (default is False).

    Returns:
    --------
    best_scores : list
        A list of the best RMSE scores achieved for each target variable.

    best_stds : list
        A list of the standard deviations corresponding to the best RMSE scores.

    summary_global : dict
        A summary dictionary for global metrics.

    summary_metrics_detailed : dict
        Updated summary dictionary for detailed metrics after the training process.
    """
    best_scores = []
    best_stds = []

    for target in targets:
        logger.info(f"Training for target: {target}")
        start_time = time.time()
        test_scores = []
        test_stds = []
        (
            X_train,
            y_train,
            X_test,
            y_test,
            df_test,
        ) = custom_train_split(
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
            summary_global, rmse, std = metrics(y_test, y_pred, summary_final, target, model_name)
            ensemble.append(model_i)
            test_scores.append(rmse)
            test_stds.append(std)

        best_scores.append(test_scores[test_scores.index(min(test_scores))])
        best_stds.append(test_stds[test_scores.index(min(test_scores))])
        logger.info("Modelisation done")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time for target {target}: {elapsed_time:.2f} seconds")

        if save:
            best_model_index = test_scores.index(min(test_scores))
            best_model = ensemble[best_model_index]

            # index_ini = df_test.index
            # df_test = df_test.merge(
            #     preprocessed_dataset[["company_id", "fiscal_year", "gics_name", "region", "country_hq"]],
            #     on=["company_id", "fiscal_year"],
            #     how="left",
            # )
            # if not restricted_features:
            #     df_test = df_test.merge(
            #         preprocessed_dataset[
            #             [
            #                 "company_id",
            #                 "fiscal_year",
            #                 "energy_consumed",
            #                 "energy_produced",
            #             ]
            #         ],
            #         on=["company_id", "fiscal_year"],
            #         how="left",
            #     )
            # df_test.index = index_ini

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
            )
    if save:
        results(estimated_scopes, path_results, summary_metrics_detailed, summary_final, lst)

    return best_scores, best_stds, summary_global, summary_metrics_detailed
