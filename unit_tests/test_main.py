import pytest
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functions.main import main
from functions.modeling import xgb, catboost, lgbm

def test_main():
    targets = ["CF1_log"]
    models = {"xgboost": xgb}
    cross_val = False
    result = main(targets, models, cross_val)
    assert result is not None