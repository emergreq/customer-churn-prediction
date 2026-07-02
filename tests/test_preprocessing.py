import numpy as np
import pandas as pd
import pytest

from customer_churn_prediction.preprocessing import (
    NEW_CLIENT_TENURE_THRESHOLD,
    add_features,
    build_preprocessor,
    clean_data,
    run_pipeline,
    split_features_target,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB", "0003-CCC", "0004-DDD"],
            "gender": ["Female", "Male", "Male", "Female"],
            "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
            "tenure": [1, 24, 5, 15],
            "MonthlyCharges": [70.0, 50.0, 30.0, 90.0],
            "TotalCharges": ["70.0", " ", "150.0", "1200.0"],
            "Churn": ["Yes", "No", "No", "Yes"],
        }
    )


def test_clean_data_fills_missing_total_charges(raw_df):
    cleaned = clean_data(raw_df)

    assert cleaned["TotalCharges"].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    # row 1 had a blank TotalCharges -> filled from tenure * MonthlyCharges
    assert cleaned.loc[1, "TotalCharges"] == pytest.approx(24 * 50.0)


def test_clean_data_does_not_mutate_input(raw_df):
    original = raw_df.copy(deep=True)
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, original)


def test_add_features_flags_new_clients(raw_df):
    result = add_features(raw_df)

    expected = (raw_df["tenure"] < NEW_CLIENT_TENURE_THRESHOLD).astype(int)
    pd.testing.assert_series_equal(result["is_new_client"], expected, check_names=False)


def test_split_features_target_encodes_churn(raw_df):
    X, y = split_features_target(raw_df)

    assert list(y) == [1, 0, 0, 1]
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns


def test_build_preprocessor_transforms_num_and_cat_columns(raw_df):
    X, _ = split_features_target(clean_data(raw_df))
    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    num_cols = X.select_dtypes(include="number").columns
    cat_cardinalities = [X[col].nunique() for col in X.select_dtypes(exclude="number").columns]
    expected_cols = len(num_cols) + sum(cat_cardinalities)

    assert transformed.shape == (len(X), expected_cols)
    assert not np.isnan(transformed).any()


def test_run_pipeline_end_to_end(tmp_path, raw_df):
    raw_path = tmp_path / "raw.csv"
    raw_df.to_csv(raw_path, index=False)

    processed_dir = tmp_path / "processed"
    preprocessor_path = tmp_path / "models" / "preprocessor.joblib"

    run_pipeline(
        raw_path=raw_path,
        processed_dir=processed_dir,
        preprocessor_path=preprocessor_path,
        test_size=0.5,
        random_state=0,
    )

    assert preprocessor_path.exists()
    assert (processed_dir / "clean_data.csv").exists()
    assert (processed_dir / "result_data.csv").exists()

    X_train = np.load(processed_dir / "X_train_processed.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_test = np.load(processed_dir / "X_test_processed.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == len(raw_df)
    assert X_train.shape[1] == X_test.shape[1]
