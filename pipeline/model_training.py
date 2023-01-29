import mlflow
from numpy import ndarray 
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.steps import step, Output, BaseParameters



experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

class ModelParameters(BaseParameters):
    name: str = "test1"
    arguments = {
        "n_estimators": 50,
        "max_depth": 20, 
        "random_state": 42
    }

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(X_train: ndarray, y_train: ndarray, parameters: ModelParameters) -> Output(model = ClassifierMixin):
    model = RandomForestClassifier(**parameters.arguments)
    mlflow.autolog()
    model.fit(X_train, y_train)
    return model