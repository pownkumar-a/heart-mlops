"""Tests for the preprocessing pipeline and model training."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import train


def test_preprocessor_fills_missing_values_and_encodes():
    X, _ = train.load_data()
    Z = train.build_preprocessor().fit_transform(X)
    if hasattr(Z, "toarray"):  # sparse matrix -> dense
        Z = Z.toarray()
    assert Z.shape[0] == len(X)
    assert Z.shape[1] > X.shape[1]  # one-hot encoding adds columns
    assert not np.isnan(Z).any()  # no missing values left


def test_pipeline_trains_and_scores_well():
    X, y = train.load_data()
    pipe = Pipeline([("prep", train.build_preprocessor()),
                     ("model", LogisticRegression(max_iter=5000))])
    pipe.fit(X, y)
    scores = train.test_scores(pipe, X, y)
    assert set(scores) == set(train.METRICS)
    assert scores["roc_auc"] > 0.8


def test_candidate_models_include_two_models():
    models = train.candidate_models()
    assert {"logistic_regression", "random_forest"} <= set(models)
