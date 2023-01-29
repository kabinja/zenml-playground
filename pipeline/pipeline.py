from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def training_rf_pipeline(
    load_data,
    feature_engineering,
    split_train_test,
    scale_training_data,
    train_model,
    evaluate_model
    ):
    data = load_data()
    data = feature_engineering(data = data)
    X_train, X_test, y_train, y_test = split_train_test(data = data)
    X_train, X_test = scale_training_data(X_train = X_train, X_test = X_test)
    model = train_model(X_train = X_train, y_train = y_train)
    recall_metric = evaluate_model(model=model, X_test = X_test, y_test = y_test)
