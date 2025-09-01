import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def sample_data():
    """
    Fixture that provides test data for all tests
    """
    # Load real data
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)
    
    # Create larger sample for better splitting
    test_sample = pd.concat([
        # Get more high earners
        data[data['salary'] == '>50K'].sample(n=8, random_state=42),
        # Get more low earners  
        data[data['salary'] == '<=50K'].sample(n=12, random_state=42),
        # Get edge cases if they exist
        data[data['workclass'] == 'Never-worked'].head(1) if not data[data['workclass'] == 'Never-worked'].empty else pd.DataFrame(),
    ]).reset_index(drop=True)
    
    return test_sample


@pytest.fixture
def categorical_features():
    """
    Fixture that provides categorical features list
    """
    return [
        "workclass",
        "education", 
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_data_types_shapes(sample_data, categorical_features):
    """
    Test that process_data function returns expected data types and shapes
    """
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    
    # Test return types
    assert isinstance(X, np.ndarray), "X should be numpy array"
    assert isinstance(y, np.ndarray), "y should be numpy array" 
    assert hasattr(encoder, 'transform'), "encoder should have transform method"
    assert hasattr(lb, 'transform'), "label binarizer should have transform method"
    
    # Test shapes
    assert X.shape[0] == len(sample_data), "X should have same number of rows as input data"
    assert y.shape[0] == len(sample_data), "y should have same number of rows as input data"
    assert len(y.shape) == 1, "y should be 1-dimensional"


def test_algorithm_type(sample_data, categorical_features):
    """
    Test that train_model returns a RandomForestClassifier instance
    """
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary", 
        training=True
    )
    
    model = train_model(X, y)
    
    # Test model type
    assert isinstance(model, RandomForestClassifier), "Model should be RandomForestClassifier"
    assert hasattr(model, 'predict'), "Model should have predict method"
    assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"
    
    # Test that model is fitted
    assert hasattr(model, 'n_features_in_'), "Model should be fitted (have n_features_in_)"


def test_model_metrics(sample_data, categorical_features):
    """
    Test that compute_model_metrics returns expected metric types and ranges
    """
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    
    model = train_model(X, y)
    preds = inference(model, X)
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # Test return types
    assert isinstance(precision, float), "Precision should be float"
    assert isinstance(recall, float), "Recall should be float"
    assert isinstance(fbeta, float), "F-beta should be float"
    
    # Test value ranges (metrics should be between 0 and 1)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1" 
    assert 0 <= fbeta <= 1, "F-beta should be between 0 and 1"


def test_inference(sample_data, categorical_features):
    """
    Test that inference function returns predictions with correct shape and type
    """
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    
    model = train_model(X, y)
    preds = inference(model, X)
    
    # Test return type and shape
    assert isinstance(preds, np.ndarray), "Predictions should be numpy array"
    assert preds.shape[0] == X.shape[0], "Predictions should have same length as input"
    assert len(preds.shape) == 1, "Predictions should be 1-dimensional"
    
    # Test prediction values (should be binary: 0 or 1)
    unique_preds = np.unique(preds)
    assert all(pred in [0, 1] for pred in unique_preds), "Predictions should be binary (0 or 1)"


def test_data_proportions(sample_data):
    """
    Test that train-test split maintains reasonable proportions
    """
    train, test = train_test_split(
        sample_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=sample_data['salary']
    )
    
    # Test sizes
    total_size = len(sample_data)
    train_size = len(train)
    test_size = len(test)
    
    assert train_size + test_size == total_size, "Train + test should equal total size"
    # Use more flexible tolerance for small samples - stratified splitting with small samples can't be exact
    assert test_size / total_size == pytest.approx(0.2, rel=0.3), "Test size should be approximately 20%"
    
    # Test data types
    assert isinstance(train, pd.DataFrame), "Train should be DataFrame"
    assert isinstance(test, pd.DataFrame), "Test should be DataFrame"
    
    # Test that all columns are preserved
    assert list(train.columns) == list(sample_data.columns), "Train should have same columns"
    assert list(test.columns) == list(sample_data.columns), "Test should have same columns"


