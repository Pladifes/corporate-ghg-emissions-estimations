import pandas as pd
import logging
import time


from functions.data_preparation import load_data
from functions.data_preparation import CarbonPricing_preprocess
from functions.data_preparation import IncomeGroup_preprocess
from functions.data_preparation import merge_datasets
from functions.data_preparation import add_variables

from functions.preprocessing import initial_preprocessing
from functions.preprocessing import custom_train_split


from functions.modeling import summary
from functions.modeling import metrics
from functions.modeling import plot
from functions.modeling import save_best_model
from functions.modeling import results
from functions.modeling import xgb, catboost, lgbm
from functions.modeling import scopes_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


def main(targets, models, cross_val):
    start_time = time.time()
    logger.info("Starting pipeline")
    ########################### PARAMETERS DEFINTION  ###########################
    path = "C:/Users/mohamed.fahmaoui/Projets/scopes_estimations/corporate_ghg_estimation/Pipeline proprietary/"
    path_rawdata = path + "data/raw_data/"
    path_models = path + "models/"
    path_Benchmark = "C:/Users/mohamed.fahmaoui/Projets/scopes_estimations/corporate_ghg_estimation/Benchmark/"
    path_results = path + "results/"
    path_mean_min = path + "data/columns_mean_min/"
    path_plot = path + "results/plot/"
    targets = ["CF1_log", "CF2_log", "CF3_log", "CF123_log"]
    models = {"xgboost": xgb, "catboost": catboost, "lgbm": lgbm}
    lst = [
        "FinalEikonID",
        "Name",
        "FiscalYear",
        "Ticker",
        "ISIN",
        "Region",
        "CountryHQ",
    ]
    summary_metrics = pd.DataFrame()
    Summary_Final = []
    ensemble = []
    test_scores = []
    best_models_lst = []
    estimated_scopes = []

    ########################### DATA LOADING  ###########################
    logger.info(f"Loading data from {path_rawdata}")
    (
        Refinitiv_data,
        CarbonPricing,
        IncomeGroup,
        FuelIntensity,
        GICSReclass,
    ) = load_data(path_rawdata)
    ########################### DATA PREPARATION  ###########################
    logger.info("Preparing data")
    CarbonPricing_Transposed = CarbonPricing_preprocess(CarbonPricing)
    logger.info("CarbonPricing preprocessed")
    IncomeGroup_Transposed = IncomeGroup_preprocess(IncomeGroup, path_rawdata)
    logger.info("IncomeGroup preprocessed")
    df_merged = merge_datasets(
        Refinitiv_data,
        GICSReclass,
        CarbonPricing_Transposed,
        IncomeGroup_Transposed,
        FuelIntensity,
    )
    logger.info("databases merged")
    raw_dataset = add_variables(df_merged)
    logger.info("financial metrics added")
    ########################### PIPELINE  ###########################
    logger.info("Running pipeline")
    for target in targets:
        dataset = initial_preprocessing(raw_dataset, target, path_rawdata)
        logger.info(f"{target} shape : {dataset.shape}")
        for i, (model_name, model) in enumerate(models.items()):
            logger.info("Train/test split done")
            X_train, y_train, X_test, y_test, features, df_test = custom_train_split(
                dataset, path_Benchmark, target, path_mean_min, path_rawdata
            )
            logger.info(f"Training model {model_name} for target {target}")
            training_time = time.time()
            model_i = model(
                X_train,
                y_train,
                cross_val=cross_val,
                n_jobs=-1,
                verbose=0,
                n_iter=10,
                seed=42,
            )
            logger.info(f"Training time taken: {time.time() - training_time} seconds.")
            y_pred = model_i.predict(X_test)
            logger.info("prediction done")
            plot(model_i, X_train, y_test, y_pred, path_plot, target, model_name)
            summary_global, rmse = metrics(
                y_test, y_pred, Summary_Final, target, model_name
            )
            logger.info(f"Model RMSE : {rmse}")

            ensemble.append(model_i)
            test_scores.append(rmse)
            metrics_scope = summary(X_test, y_pred, y_test, df_test, target, model_name)
            summary_metrics = pd.concat(
                [summary_metrics, metrics_scope], ignore_index=True
            )
        best_model, best_models_lst = save_best_model(
            best_models_lst, test_scores, ensemble, path_models, target, model_name
        )
        estimated_scopes = scopes_report(
            dataset, features, target, best_model, estimated_scopes, lst, path_results
        )
        logger.info("detailed metrics for Sectors, Subsectors,region report generated")

    results(
        estimated_scopes,
        path_results,
        summary_metrics,
        summary_global,
        lst,
        name_experiment,
    )
    logger.info(f"Total time taken: {time.time() - start_time} seconds.")
    logger.info("dashboard generated")
    logger.info("reports generated")
