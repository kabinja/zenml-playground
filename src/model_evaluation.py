import mlflow

from numpy import ndarray
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from zenml.steps import step, Output
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def evaluate_model(model: ClassifierMixin, X_test: ndarray, y_test: ndarray) -> Output(
    accuracy = float, precision = float, recall = float, f1 = float
    ):
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average = 'macro')
    recall = recall_score(y_test, y_preds, average = 'macro')
    f1 = f1_score(y_test, y_preds, average = 'macro')
    mse = mean_squared_error(y_test, y_preds)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("mse", mse)
    signature = _specify_signature()
    mlflow.sklearn.log_model(model, "random_forest", signature=signature)
    return accuracy, precision, recall, f1


def _specify_signature() -> ModelSignature:
    input_schema = Schema(
        [
            ColSpec("integer", "Age (days)"),
            ColSpec("integer", "Gender (categorical code)"),
            ColSpec("integer", "Height (cm)"),
            ColSpec("float", "Weight (kg)"),
            ColSpec("integer", "Systolic blood pressure"),
            ColSpec("integer", "Diastolic blood pressure"),
            ColSpec("integer", "Cholesterol ( 1: normal, 2: above normal, 3: well above normal)"),
            ColSpec("integer", "Smoking (binary)"),
            ColSpec("integer", "Alcohol intake (binary)"),
            ColSpec("integer", "Glucose (1: normal, 2: above normal, 3: well above normal)"),
            ColSpec("integer", "Physical activity (binary)"),
        ]
    )
    output_schema = Schema([ColSpec("integer", "Presence or absence of cardiovascular disease (binary)")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)