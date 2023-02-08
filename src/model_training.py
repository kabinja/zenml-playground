import mlflow
from numpy import ndarray 
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.steps import step, Output, BaseParameters

class ModelParameters(BaseParameters):
    name: str = "test1"
    arguments = {
        "n_estimators": 50,
        "max_depth": 20, 
        "random_state": 42
    }

@step(enable_cache=False)
def train_model(X_train: ndarray, y_train: ndarray, parameters: ModelParameters) -> Output(model = ClassifierMixin):
    model = RandomForestClassifier(**parameters.arguments)
    model.fit(X_train, y_train)
    return model