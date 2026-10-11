"""Tests for loading the dataset."""
import pandas as pd

from src import train


def test_dataset_has_expected_shape():
    df = pd.read_csv(train.DATA_PATH)
    assert df.shape == (303, 14)


def test_load_data_returns_features_and_binary_target():
    X, y = train.load_data()
    assert list(X.columns) == train.FEATURES
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y) == 303


def test_missing_values_only_in_ca_and_thal():
    X, _ = train.load_data()
    columns_with_missing = set(X.columns[X.isna().any()])
    assert columns_with_missing <= {"ca", "thal"}
