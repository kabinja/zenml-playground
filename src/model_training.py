import mlflow
from numpy import ndarray 
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from zenml.steps import step, Output, BaseParameters

class ModelParameters(BaseParameters):
    name: str = "test1"
    arguments = {
        "n_estimators": 6,
        "max_depth": 3, 
        "random_state": 42
    }

@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model(X_train: ndarray, y_train: ndarray, parameters: ModelParameters) -> Output(model = ClassifierMixin):
    mlflow.log_params(parameters.arguments)
    model = RandomForestClassifier(**parameters.arguments)
    model.fit(X_train, y_train)
    return model