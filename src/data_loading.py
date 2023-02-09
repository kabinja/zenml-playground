from io import StringIO
import pandas

from zenml.steps import step, Output
from pandas import DataFrame
import urllib

@step
def load_data() -> Output(data = DataFrame):
    response = urllib.request.urlopen("https://raw.githubusercontent.com/kabinja/zenml-playground/main/data/cardio_train.csv")
    csv_content = StringIO(str(response.read(),'utf-8'))
    return pandas.read_csv(csv_content, sep=";")