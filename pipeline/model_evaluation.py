import mlflow

from numpy import ndarray
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from zenml.steps import step, Output

@step(enable_cache=False)
def evaluate_model(model: ClassifierMixin, X_test: ndarray, y_test: ndarray) -> Output(
    accuracy = float, precision = float, recall = float, f1 = float
    ):
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average = 'macro')
    recall = recall_score(y_test, y_preds, average = 'macro')
    f1 = f1_score(y_test, y_preds, average = 'macro')
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    return accuracy, precision, recall, f1