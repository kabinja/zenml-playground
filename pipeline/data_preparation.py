from numpy import ndarray, int16, int8
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from zenml.steps import step, Output

@step
def feature_engineering(data: DataFrame) -> Output(dataframe=DataFrame):
    dataframe = data.copy()
    dataframe['age'] = dataframe['age'].astype(int16)
    dataframe['ap_hi'] = dataframe['ap_hi'].astype(int16)
    dataframe['ap_lo'] = dataframe['ap_lo'].astype(int16)
    dataframe['cholesterol'] = dataframe['cholesterol'].astype(int8)
    dataframe['cardio'] = dataframe['cardio'].astype(int8)
    dataframe['age'] = round(dataframe['age'] / 365).astype(int)
    dataframe = dataframe[dataframe['weight'] >= 33]
    dataframe = dataframe[(dataframe['ap_hi'] >=85) & (dataframe['ap_hi'] <= 240)]
    dataframe = dataframe[(dataframe['ap_lo'] >=65) & (dataframe['ap_lo'] <= 150)]
    return dataframe


@step
def split_train_test(data: DataFrame) -> Output(
    X_train = ndarray, X_test = ndarray, y_train = ndarray, y_test = ndarray
    ):
    X = data.drop('cardio', axis = 1).values
    y = data['cardio'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@step
def scale_training_data(X_train: ndarray, X_test: ndarray) -> Output(
    X_train_scaled = ndarray, X_test_scaled = ndarray
    ):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled