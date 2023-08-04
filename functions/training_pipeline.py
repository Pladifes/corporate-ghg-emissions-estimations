import mlflow
import numpy as NPPE

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from functions.preprocessing import custom_train_split
from functions.results import best_model_analysis, metrics, results


def weights_creation(df, target, companies=True):
    df["weight_reliability"] = np.ones(len(df))
    CDP_indexes = df[df[target] == df["CDP_"+target]].index
    df.loc[CDP_indexes, "weight_reliability"] = [2 for i in range(len(CDP_indexes))]

    nb_occurences = df["country_sector"].value_counts() 
    df["weight_country_sector"] = df.apply(lambda row : 1 / nb_occurences[row["country_sector"]])

    if companies:
        nb_occurences = df.FinalEikonID.value_counts()
        df["weight_companies"] = df.apply(lambda row : 1 / nb_occurences[row["FinalEikonID"]])

        df["weight_final"] = df["weight_reliability"] * df["weight_companies"] * df["weight_country_sector"]
    else:
        df["weight_final"] = df["weight_reliability"] *  df["weight_country_sector"]

    return df["weight_final"].tolist()


def training_pipeline(
    name_experiment,
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
    open_data=False,
    save=False,
    weights=None,
    companies=True,
):
    """
    Apply a training pipeline for the imputes targets, models and parameters.
    """
    best_scores = []
    best_stds = []
    mlflow.create_experiment("" f"Models_{name_experiment}")
    mlflow.set_experiment("" f"Models_{name_experiment}")

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
            # threshold_under=training_parameters["threshold_under"],
            # threshold_over=training_parameters["threshold_over"],
            extended_features=training_parameters["extended_features"],
            selec_sect=training_parameters["selec_sect"],
            fill_grp=training_parameters["fill_grp"],
            old_pipe=training_parameters["old_pipe"],
            open_data=open_data,
        )
        print("preprocessing done")
        if weights:
            df_train_merged = X_train.join(preprocessed_dataset[["CDP_CF1", "CDP_CF2", "CDP_CF3", "CDP_CF123","country_sector"]]) # (11056, 281)        
            weights = weights_creation(df_train_merged, target, companies)

        seed = training_parameters["seed"]
        n_iter = training_parameters["n_iter"]
        for i, (model_name, model) in enumerate(models.items()):
            with mlflow.start_run() as _:
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
                mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
                mlflow.log_metric("rmse", mean_squared_error(y_test, y_pred, squared=False))
                mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
                mlflow.log_metric("r2", r2_score(y_test, y_pred))
                mlflow.log_metric("mape", mean_absolute_percentage_error(y_test, y_pred))
                mlflow.log_param("target", target)
                mlflow.log_param("model", model_name)
                mlflow.sklearn.log_model(model, "models", registered_model_name=model_name)
                ensemble.append(model_i)
                # model_name_lst.append(model_name)
                test_scores.append(rmse)
                test_stds.append(std)

        best_scores.append(test_scores[test_scores.index(min(test_scores))])
        best_stds.append(test_stds[test_scores.index(min(test_scores))])
        print("modelisation done")

        if save:
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
                training_parameters,
                open_data,
                path_models,
            )

            results(estimated_scopes, path_results, summary_metrics_detailed, Summary_Final, lst)

    return best_scores, best_stds, summary_global, summary_metrics_detailed
