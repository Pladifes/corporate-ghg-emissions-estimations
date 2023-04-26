import pandas as pd
import numpy as np
import pytest
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.data_preparation import load_data
from functions.data_preparation import CarbonPricing_preprocess
from functions.data_preparation import IncomeGroup_preprocess
from functions.data_preparation import merge_datasets
from functions.data_preparation import add_variables

from functions.preprocessing import initial_preprocessing
from functions.preprocessing import custom_train_split

from functions.modeling import summary
from functions.modeling import plot
from functions.modeling import xgb
from functions.modeling import scopes_report
from functions.modeling import metrics

def create_dataset_and_model():
    path_rawdata = 'data/raw_data/'
    path_Benchmark = '../Benchmark/'
    path_mean_min = 'data/columns_mean_min/'
    path_plot = 'results/plot/'
    targets = ["CF1_log"]
    models = {
            "xgboost": xgb}
    lst = ['FinalEikonID','Name','FiscalYear','Ticker','ISIN', 'Region','CountryHQ']
    Refinitiv_data, CarbonPricing, IncomeGroup, FuelIntensity,GICSReclass = load_data(path_rawdata) 
    CarbonPricing_Transposed = CarbonPricing_preprocess(CarbonPricing)
    IncomeGroup_Transposed = IncomeGroup_preprocess(IncomeGroup,path_rawdata)
    df_merged =merge_datasets(
                            Refinitiv_data,
                            GICSReclass,
                            CarbonPricing_Transposed,
                            IncomeGroup_Transposed,
                            FuelIntensity,
                            )
    raw_dataset = add_variables(df_merged)
    dataset = initial_preprocessing(raw_dataset, targets,path_rawdata)
    X_train, y_train, X_test, y_test,features,df_test = custom_train_split(dataset, path_Benchmark, targets, path_mean_min,path_rawdata)                  
    model_i = models.item()(X_train, y_train, cross_val=False, n_jobs=-1, verbose=0,n_iter=10, seed=42)
    y_pred = model_i.predict(X_test)
    return X_train, y_train, X_test, y_test,features,df_test,y_pred,model_i,dataset

def test_summary():
    X_train, y_train, X_test, y_test,features,df_test,y_pred,model_i,dataset = create_dataset_and_model()
    result = summary(X_test, y_pred, y_test, df_test, 'CF1_log', model_i)

    assert all(result.columns == ['category', 'scope', 'model_name', 'category_name', 'RMSE', 'MSE', 'MAE', 'MAPE', 'R2', 'StandardDeviation'])
    assert result.loc[0, 'category'] == 'SectorName'
    assert result.loc[0, 'scope'] == 'CF1_log'
    assert result.loc[0, 'model_name'] == 'xgboost'

def test_plot():
    X_train, y_train, X_test, y_test,features,df_test,y_pred,model_i,dataset = create_dataset_and_model()
    plot_path = 'results/plot/'
    plot(model_i, X_test, y_test, y_pred, plot_path, 'CF1_log', 'xgboost')
    assert os.path.exists(plot_path+f'shap_CF1_log_xgboost.png')
    assert os.path.exists(plot_path+f'y_test_y_pred_CF1_log_xgboost.png')
    assert os.path.exists(plot_path+f'residus_CF1_log_xgboost.png')

def test_scopes_report():
    path_results = 'results/'
    lst = ['FinalEikonID','Name','FiscalYear','Ticker','ISIN', 'Region','CountryHQ']
    X_train, y_train, X_test, y_test,features,df_test,y_pred,model_i,dataset = create_dataset_and_model()
    estimated_scopes = scopes_report(dataset, features, 'CF1_log', model_i, estimated_scopes, lst, path_results)
    assert len(estimated_scopes) == len(dataset)
   

def test_metrics():
    Summary_Final = []
    X_train, y_train, X_test, y_test,features,df_test,y_pred,model_i,dataset = create_dataset_and_model()
    summary_global, rmse = metrics(y_test, y_pred, Summary_Final, 'CF1_log', 'xgboost')

    assert isinstance(summary_global, pd.DataFrame)
    assert summary_global.iloc[0]["Target"] == 'CF1_log'
    assert summary_global.iloc[0]["model"] == 'xgboost'

def model_data():
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([10, 20, 30])
    return X_train, y_train

def test_xgb(model_data):
    X_train, y_train = model_data
    model = xgb(X_train, y_train)
    assert model.get_booster().best_iteration > 0
