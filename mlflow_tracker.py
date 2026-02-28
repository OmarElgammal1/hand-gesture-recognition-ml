import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_mlflow_experiment(models, X_train, X_test, y_train, y_test,
                          experiment_name="Hand_Gesture_Classification"):
    """
    Trains and evaluates a dictionary of sklearn-compatible models,
    logging each run to MLflow.

    Args:
        models (dict): Mapping of model name (str) -> unfitted sklearn estimator.
        X_train, X_test (array-like): Feature splits.
        y_train, y_test (array-like): Label splits.
        experiment_name (str): MLflow experiment name. Defaults to
            'Hand_Gesture_Classification'.

    Returns:
        pd.DataFrame: Summary table with Accuracy, Precision, Recall, F1-Score,
            sorted by Accuracy descending.
    """
    mlflow.set_experiment(experiment_name)
    print(f"Starting MLflow experiment: '{experiment_name}'\n")

    summary_results = []

    for model_name, clf in models.items():
        print(f"  Training {model_name}...")
        with mlflow.start_run(run_name=model_name):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall",    rec)
            mlflow.log_metric("f1_score",  f1)
            mlflow.sklearn.log_model(clf, "model")

            summary_results.append({
                "Model Name": model_name,
                "Accuracy":   acc,
                "Precision":  prec,
                "Recall":     rec,
                "F1-Score":   f1,
            })

    summary_df = (
        pd.DataFrame(summary_results)
        .sort_values(by="Accuracy", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- MLflow Evaluation Summary ---")
    print(summary_df.to_string(index=False))
    print("\n✅ All models and metrics have been logged to MLflow.")

    return summary_df
