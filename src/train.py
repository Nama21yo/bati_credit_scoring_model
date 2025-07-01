import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from src.data_processing import process_data, build_pipeline


def evaluate_model(y_true, y_pred, y_prob):
    """Evaluate model performance with multiple metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    return metrics


def train_models():
    """Train and evaluate models, logging to MLflow."""
    df = process_data(save=False)
    X = df.drop(["CustomerId", "is_high_risk", "last_txn_time"], axis=1)
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    with mlflow.start_run(run_name="credit_risk_training"):
        # Logistic Regression
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train_processed, y_train)
        lr_pred = lr.predict(X_test_processed)
        lr_prob = lr.predict_proba(X_test_processed)[:, 1]
        lr_metrics = evaluate_model(y_test, lr_pred, lr_prob)
        mlflow.log_metrics({f"lr_{k}": v for k, v in lr_metrics.items()})
        mlflow.sklearn.log_model(lr, "logistic_regression")

        # Random Forest with Grid Search
        rf = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [50, 100], "max_depth": [10, 20]}
        rf_grid = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc")
        rf_grid.fit(X_train_processed, y_train)
        rf_pred = rf_grid.predict(X_test_processed)
        rf_prob = rf_grid.predict_proba(X_test_processed)[:, 1]
        rf_metrics = evaluate_model(y_test, rf_pred, rf_prob)
        mlflow.log_metrics({f"rf_{k}": v for k, v in rf_metrics.items()})
        mlflow.sklearn.log_model(rf_grid.best_estimator_, "random_forest")

        # Register best model (based on ROC-AUC)
        if rf_metrics["roc_auc"] > lr_metrics["roc_auc"]:
            best_model = rf_grid.best_estimator_
            model_name = "random_forest"
        else:
            best_model = lr
            model_name = "logistic_regression"
        mlflow.sklearn.log_model(best_model, model_name)
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/{model_name}", "CreditRiskModel"
        )


if __name__ == "__main__":
    train_models()
