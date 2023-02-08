import pandas

from zenml.steps import step, Output
from pandas import DataFrame

@step
def load_data() -> Output(data = DataFrame):
    return pandas.read_csv('data/cardio_train.csv', sep=";")