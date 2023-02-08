from model_evaluation import evaluate_model
from model_training import train_model
from pipeline import training_rf_pipeline
from data_loading import load_data
from data_preparation import feature_engineering, scale_training_data, split_train_test

def run_pipeline():
    training = training_rf_pipeline(
        load_data = load_data(),
        feature_engineering = feature_engineering(),
        split_train_test = split_train_test(),
        scale_training_data = scale_training_data(),
        train_model = train_model(),
        evaluate_model = evaluate_model(),
    )  
    training.run()


if __name__ == "__main__":
    run_pipeline()