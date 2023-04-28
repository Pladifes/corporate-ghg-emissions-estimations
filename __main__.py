
from functions.production_pipeline import production_pipeline
from functions.models import xgboost_model, catboost_model, lgbm_model

name_experiment = "test_main"
targets = ["CF1_log", "CF2_log"]
models = {
        # "xgboost": xgboost_model,
        "catboost": catboost_model,
        "lgbm": lgbm_model,
    }
open_data = False

if __name__ == "__main__":
    production_pipeline(targets= targets, models = models, open_data=open_data, name_experiment=name_experiment)