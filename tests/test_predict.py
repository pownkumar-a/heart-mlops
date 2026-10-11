"""Tests for the prediction function."""
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import train
from src.predict import SAMPLE_PATIENT, predict


@pytest.fixture(scope="module")
def model():
    X, y = train.load_data()
    pipe = Pipeline([("prep", train.build_preprocessor()),
                     ("model", LogisticRegression(max_iter=5000))])
    return pipe.fit(X, y)


def test_predict_returns_prediction_and_confidence(model):
    result = predict([SAMPLE_PATIENT], model=model)[0]
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("heart disease", "no heart disease")
    assert 0.5 <= result["confidence"] <= 1.0
    assert 0.0 <= result["disease_probability"] <= 1.0


def test_predict_handles_missing_values(model):
    patient = dict(SAMPLE_PATIENT, ca=None, thal=None)
    assert predict([patient], model=model)[0]["prediction"] in (0, 1)


def test_predict_batch_of_patients(model):
    assert len(predict([SAMPLE_PATIENT] * 3, model=model)) == 3
