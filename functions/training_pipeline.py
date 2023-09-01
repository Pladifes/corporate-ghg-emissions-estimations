from functions.preprocessing import custom_train_split
from functions.results import best_model_analysis, metrics, results


def training_pipeline(
    path_Benchmark,
    path_results,
    path_models,
    path_intermediary,
    path_plot,
    targets,
    models,
    Summary_Final,
    ensemble,
    summary_metrics_detailed,
    estimated_scopes,
    preprocessed_dataset,
    training_parameters,
    restricted_features=False,
    save=False,
):
    """
    Apply a training pipeline for the imputes targets, models and parameters.
    """
    best_scores = []
    best_stds = []

    for target in targets:
        print(target)
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
            path_Benchmark,
            path_intermediary,
            target,
            extended_features=training_parameters["extended_features"],
            selec_sect=training_parameters["selec_sect"],
            restricted_features=restricted_features,
        )
        print("preprocessing done")

        seed = training_parameters["seed"]
        n_iter = training_parameters["n_iter"]
        for i, (model_name, model) in enumerate(models.items()):
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
            summary_global, rmse, std = metrics(y_test, y_pred, Summary_Final, target, model_name)
            ensemble.append(model_i)
            test_scores.append(rmse)
            test_stds.append(std)

        best_scores.append(test_scores[test_scores.index(min(test_scores))])
        best_stds.append(test_stds[test_scores.index(min(test_scores))])
        print("modelisation done")

        if save:
            best_model_index = test_scores.index(min(test_scores))
            best_model = ensemble[best_model_index]
            if not restricted_features:
                index_ini = df_test.index
                df_test = df_test.merge(
                    preprocessed_dataset[["FinalEikonID", "FiscalYear", "ENEConsume", "ENEProduce"]],
                    on=["FinalEikonID", "FiscalYear"],
                    how="left",
                )
                df_test.index = index_ini

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
        results(estimated_scopes, path_results, summary_metrics_detailed, Summary_Final, lst)

    return best_scores, best_stds, summary_global, summary_metrics_detailed
