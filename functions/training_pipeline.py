import mlflow
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from functions.preprocessing import custom_train_split
from functions.results import best_model_analysis, metrics, results


def companies_occurences(df):
    nb_occurences = df.FinalEikonID.value_counts()
    return df.apply(lambda row: 1 / nb_occurences[row["FinalEikonID"]], axis=1).values


def special(df, target):
    if target == "CF2":
        index_to_scale = df[(df[target + "_log"] < 4)].index
        df.loc[index_to_scale, "weight_final"] = df["weight_final"] * 20
    elif target == "CF3":
        index_to_scale = df[(df[target + "_log"] < 3) | (df[target + "_log"] > 6)].index
        df.loc[index_to_scale, "weight_final"] = df["weight_final"] * 20
    elif target == "CF123":
        index_to_scale = df[(df[target + "_log"] > 5) & (df[target + "_log"])].index
        df.loc[index_to_scale, "weight_final"] = df["weight_final"] * 10
        index_to_scale = df[(df[target + "_log"] > 7) & (df[target + "_log"])].index
        df.loc[index_to_scale, "weight_final"] = df["weight_final"] * 100
    return df


def weights_creation(df, target, companies=True):
    target = target[:-4]
    df["weight_reliability"] = np.ones(len(df))
    CDP_indexes = df[df[target] == df["CDP_" + target]].index
    df.loc[CDP_indexes, "weight_reliability"] = [2 for i in range(len(CDP_indexes))]

    nb_occurences = df["country_sector"].value_counts()
    df["weight_country_sector"] = df.apply(lambda row: 1 / nb_occurences[row["country_sector"]], axis=1)

    if companies:
        df["weight_companies"] = companies_occurences(df)
        df["weight_final"] = df["weight_reliability"] * df["weight_companies"] * df["weight_country_sector"]
    else:
        df["weight_final"] = df["weight_reliability"]  # * df["weight_country_sector"]

    df = special(df, target)
    return np.array(df["weight_final"].tolist())


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
    use_weights=False,
    companies=False,
    custom_gradient=False,
):
    """
    Apply a training pipeline for the imputes targets, models and parameters.
    """
    best_scores = []
    best_stds = []

    # Only to test CF3_E = CF123_E - CF1_E - CF2_E
    # store_pred = []
    # store_test = []

    mlflow.create_experiment("" f"Models_{name_experiment}")
    mlflow.set_experiment("" f"Models_{name_experiment}")

    for target in targets:
        print(target)
        test_scores = []
        test_stds = []
        # test_scores_train = []
        # test_stds_train = []
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
            open_data=open_data,
        )
        print("preprocessing done")
        if use_weights or custom_gradient:
            df_train_merged = X_train.join(
                preprocessed_dataset[
                    [
                        "CF1",
                        "CF2",
                        "CF3",
                        "CF123",
                        "CDP_CF1",
                        "CDP_CF2",
                        "CDP_CF3",
                        "CDP_CF123",
                        "country_sector",
                        "FinalEikonID",
                    ]
                ]
            )
            df_train_merged[target] = y_train
            if custom_gradient:
                weights = companies_occurences(df_train_merged)
            else:
                weights = weights_creation(df_train_merged, target, companies)
        else:
            weights = None

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
                    weights=weights,
                    custom_gradient=custom_gradient,
                )
                y_pred = model_i.predict(X_test)
                # y_pred_train = model_i.predict(X_train)

                # store_pred.append(y_pred)
                # store_test.append(y_test)

                summary_global, rmse, std = metrics(y_test, y_pred, Summary_Final, target, model_name)
                # summary_global_train, rmse_train, std_train = metrics(
                #     y_train, y_pred_train, Summary_Final, target, model_name
                # )
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

                # test_scores_train.append(rmse_train)
                # test_stds_train.append(std_train)

        best_scores.append(test_scores[test_scores.index(min(test_scores))])
        best_stds.append(test_stds[test_scores.index(min(test_scores))])

        # best_scores.append(test_scores_train[test_scores_train.index(min(test_scores_train))])
        # best_stds.append(test_stds_train[test_scores_train.index(min(test_scores_train))])

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
                open_data,
                path_models,
            )
    if save:
        results(estimated_scopes, path_results, summary_metrics_detailed, Summary_Final, lst)

    # Only to test CF3_E = CF123_E - CF1_E - CF2_E
    # summary_global, rmse, std = metrics(
    #     store_test[2], store_pred[3] - store_pred[0] - store_pred[1], Summary_Final, "CF3bis", "models difference"
    # )
    # best_scores.append(rmse)
    # best_stds.append(std)

    return best_scores, best_stds, summary_global, summary_metrics_detailed
