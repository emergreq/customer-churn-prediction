import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

from customer_churn_prediction.preprocessing import add_features, clean_data

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
IMAGES_DIR = Path("images")

RANDOM_STATE = 42

# Diverging pair used for polarity (risk vs. protective effect), matches project's other plots.
RISK_COLOR = "#e34948"
PROTECTIVE_COLOR = "#2a78d6"


def load_processed_data(processed_dir: Path = PROCESSED_DIR):
    X_train = np.load(processed_dir / "X_train_processed.npy")
    X_test = np.load(processed_dir / "X_test_processed.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "report": classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
        "report_dict": classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True
        ),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }


def train_and_evaluate(processed_dir: Path = PROCESSED_DIR, models_dir: Path = MODELS_DIR) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    results = {}
    fitted_models = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        print(f"=== {name} ===")
        print(metrics["report"])
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"PR-AUC:  {metrics['pr_auc']:.3f}\n")

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = fitted_models[best_name]

    model_path = models_dir / "churn_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Лучшая модель по ROC-AUC: {best_name} -> сохранена в {model_path}")

    metrics_path = models_dir / "metrics.json"
    metrics_to_save = {
        name: {"roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"], "report": r["report_dict"]}
        for name, r in results.items()
    }
    metrics_to_save["best_model"] = best_name
    metrics_path.write_text(json.dumps(metrics_to_save, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Метрики сохранены в {metrics_path}")

    return results


def get_feature_importance(model, preprocessor) -> pd.DataFrame:
    if not hasattr(model, "coef_"):
        raise TypeError(
            f"{type(model).__name__} does not expose coefficients; "
            "use feature_importances_ or SHAP for tree-based models instead"
        )

    feature_names = pd.Series(preprocessor.get_feature_names_out())
    feature_names = feature_names.str.replace(r"^(num|cat)__", "", regex=True)

    importance = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_[0]})
    importance["odds_ratio"] = np.exp(importance["coefficient"])
    return importance.sort_values("coefficient", ascending=False).reset_index(drop=True)


def plot_feature_importance(
    importance: pd.DataFrame, top_n: int = 8, images_dir: Path = IMAGES_DIR
) -> Path:
    images_dir.mkdir(parents=True, exist_ok=True)
    top = pd.concat([importance.head(top_n), importance.tail(top_n)]).sort_values("coefficient")
    colors = [RISK_COLOR if c > 0 else PROTECTIVE_COLOR for c in top["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_xlabel("Коэффициент логистической регрессии (лог-шансы)")
    ax.set_title("Что сильнее всего влияет на отток")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    path = images_dir / "model_coefficients.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def interpret_best_model(
    models_dir: Path = MODELS_DIR, images_dir: Path = IMAGES_DIR
) -> pd.DataFrame:
    model = joblib.load(models_dir / "churn_model.joblib")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")

    importance = get_feature_importance(model, preprocessor)
    image_path = plot_feature_importance(importance, images_dir=images_dir)

    print("Топ факторов риска (увеличивают отток):")
    print(importance.head(8).to_string(index=False))
    print("\nТоп защитных факторов (снижают отток):")
    print(importance.tail(8).sort_values("coefficient").to_string(index=False))
    print(f"\nГрафик сохранён в {image_path}")

    return importance


def tune_logistic_regression(
    processed_dir: Path = PROCESSED_DIR, models_dir: Path = MODELS_DIR
) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "l1_ratio": [0, 1],  # 0 = l2, 1 = l1
    }
    base_model = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=1000, random_state=RANDOM_STATE
    )
    search = GridSearchCV(base_model, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"Лучшие параметры: {search.best_params_}")
    print(f"CV ROC-AUC: {search.best_score_:.3f}")

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)
    print(metrics["report"])
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Test PR-AUC:  {metrics['pr_auc']:.3f}")

    model_path = models_dir / "churn_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Модель с подобранными гиперпараметрами сохранена в {model_path}")

    metrics_path = models_dir / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    all_metrics["logistic_regression_tuned"] = {
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "report": metrics["report_dict"],
        "best_params": search.best_params_,
        "cv_roc_auc": float(search.best_score_),
    }
    all_metrics["best_model"] = "logistic_regression_tuned"
    metrics_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics


def predict_churn(
    customer_data: dict | list[dict] | pd.DataFrame, models_dir: Path = MODELS_DIR
) -> pd.DataFrame:
    if isinstance(customer_data, pd.DataFrame):
        df = customer_data.copy()
    else:
        rows = [customer_data] if isinstance(customer_data, dict) else customer_data
        df = pd.DataFrame(rows)

    df = clean_data(df)
    df = add_features(df)
    X = df.drop(columns=["Churn", "customerID", "Unnamed: 0"], errors="ignore")

    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    model = joblib.load(models_dir / "churn_model.joblib")
    churn_proba = model.predict_proba(preprocessor.transform(X))[:, 1]

    result = df[["customerID"]].copy() if "customerID" in df.columns else pd.DataFrame(index=df.index)
    result["churn_probability"] = churn_proba
    result["churn_prediction"] = np.where(churn_proba >= 0.5, "Yes", "No")
    return result


if __name__ == "__main__":
    train_and_evaluate()
    interpret_best_model()
