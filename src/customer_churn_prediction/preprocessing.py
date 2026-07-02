from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = Path("data/Telco-Customer-Churn.csv")
PROCESSED_DIR = Path("data/processed")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

NEW_CLIENT_TENURE_THRESHOLD = 12


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_new_client"] = (df["tenure"] < NEW_CLIENT_TENURE_THRESHOLD).astype(int)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn", "customerID", "Unnamed: 0"], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(exclude="number").columns.to_list()
    num_cols = X.select_dtypes(include="number").columns.to_list()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def run_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    processed_dir: Path = PROCESSED_DIR,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    df = clean_data(df)
    df.to_csv(processed_dir / "clean_data.csv", index=False)

    df = add_features(df)
    df.to_csv(processed_dir / "result_data.csv", index=False)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(X)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, preprocessor_path)
    np.save(processed_dir / "X_train_processed.npy", X_train_processed)
    np.save(processed_dir / "X_test_processed.npy", X_test_processed)
    np.save(processed_dir / "y_train.npy", y_train.values)
    np.save(processed_dir / "y_test.npy", y_test.values)


if __name__ == "__main__":
    run_pipeline()
