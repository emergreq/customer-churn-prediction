import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from customer_churn_prediction.models import evaluate_model, get_feature_importance, predict_churn
from customer_churn_prediction.preprocessing import (
    build_preprocessor,
    clean_data,
    split_features_target,
)


@pytest.fixture
def toy_classification_data():
    X, y = make_classification(
        n_samples=200, n_features=6, weights=[0.7, 0.3], random_state=0
    )
    return X, y


def test_evaluate_model_returns_expected_metrics(toy_classification_data):
    X, y = toy_classification_data
    model = LogisticRegression().fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert "Churn" in metrics["report"]
    assert "Churn" in metrics["report_dict"]


def test_get_feature_importance_matches_coefficients(toy_classification_data):
    X, y = toy_classification_data
    model = LogisticRegression().fit(X, y)

    class IdentityPreprocessor:
        def get_feature_names_out(self):
            return np.array([f"num__f{i}" for i in range(X.shape[1])])

    importance = get_feature_importance(model, IdentityPreprocessor())

    assert len(importance) == X.shape[1]
    assert set(importance["feature"]) == {f"f{i}" for i in range(X.shape[1])}
    assert importance["coefficient"].is_monotonic_decreasing
    np.testing.assert_allclose(importance["odds_ratio"], np.exp(importance["coefficient"]))


def test_get_feature_importance_rejects_models_without_coefficients(toy_classification_data):
    X, y = toy_classification_data
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    with pytest.raises(TypeError):
        get_feature_importance(model, preprocessor=None)


@pytest.fixture
def fitted_artifacts(tmp_path):
    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-AAA" for i in range(20)],
            "gender": ["Female", "Male"] * 10,
            "Contract": ["Month-to-month", "Two year"] * 10,
            "tenure": list(range(1, 21)),
            "MonthlyCharges": [50.0 + i for i in range(20)],
            "TotalCharges": [str(50.0 + i) for i in range(20)],
            "Churn": ["Yes", "No"] * 10,
        }
    )
    df = clean_data(df)
    X, y = split_features_target(df)

    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    model = LogisticRegression().fit(X_processed, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(model, models_dir / "churn_model.joblib")

    return models_dir


def test_predict_churn_on_single_new_customer(fitted_artifacts):
    new_customer = {
        "gender": "Female",
        "Contract": "Month-to-month",
        "tenure": 2,
        "MonthlyCharges": 55.0,
        "TotalCharges": "110.0",
    }

    result = predict_churn(new_customer, models_dir=fitted_artifacts)

    assert len(result) == 1
    assert 0.0 <= result.loc[0, "churn_probability"] <= 1.0
    assert result.loc[0, "churn_prediction"] in {"Yes", "No"}


def test_predict_churn_on_multiple_customers_keeps_customer_id(fitted_artifacts):
    new_customers = pd.DataFrame(
        {
            "customerID": ["9001-XYZ", "9002-XYZ"],
            "gender": ["Female", "Male"],
            "Contract": ["Month-to-month", "Two year"],
            "tenure": [2, 30],
            "MonthlyCharges": [55.0, 65.0],
            "TotalCharges": ["110.0", "1950.0"],
        }
    )

    result = predict_churn(new_customers, models_dir=fitted_artifacts)

    assert list(result["customerID"]) == ["9001-XYZ", "9002-XYZ"]
    assert result["churn_probability"].between(0, 1).all()
