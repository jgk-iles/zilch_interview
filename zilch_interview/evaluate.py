from dvclive import Live
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)

import os
import argparse
import pickle as pkl
import pandas as pd


def evaluate(model, data, split, live):

    data_scored = data.copy()

    try:
        X = data.drop(columns=["credit_score_target"])
        y = data["credit_score_target"]
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)

        # Log the evaluation metrics
        if not live.summary:
            live.summary = {"mae": {}, "mse": {}, "rmse": {}}

        live.summary["mae"][split] = mae
        live.summary["mse"][split] = mse
        live.summary["rmse"][split] = rmse

        data_scored["credit_score_prediction"] = y_pred.astype(int)
        return data_scored

    except KeyError:
        X = data.copy()
        y_pred = model.predict(X)
        data_scored["credit_score_prediction"] = y_pred.astype(int)
        return data_scored


def main():
    EVAL_DIR = "evaluation"
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Get arguments from the command line
    parser = argparse.ArgumentParser(description="Model evaluation pipeline step")
    parser.add_argument("--model", type=str, help="Path to the trained model")
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    parser.add_argument("--test_data", type=str, help="Path to the test data")
    parser.add_argument("--validation_data", type=str, help="Path to the validation data")
    args = parser.parse_args()

    # Load the model and data
    with open(args.model, "rb") as model_file:
        model = pkl.load(model_file)

    train_data = pd.read_pickle(args.train_data)
    test_data = pd.read_pickle(args.test_data)
    validation_data = pd.read_pickle(args.validation_data)

    with Live(EVAL_DIR) as live:
        train_scored = evaluate(model, train_data, "train", live)
        test_scored = evaluate(model, test_data, "test", live)
        validation_scored = evaluate(model, validation_data, "validation", live)
        
    # Load customer ids from the raw data and append to the scored data
    train_raw = pd.read_csv("data/external/train.csv")
    test_raw = pd.read_csv("data/external/test.csv")
    validation_raw = pd.read_csv("data/external/validation.csv")
    
    train_scored["customer_id"] = train_raw["customer_id"]
    test_scored["customer_id"] = test_raw["customer_id"]
    validation_scored["customer_id"] = validation_raw["customer_id"]
    
    train_scored = train_scored[["customer_id"] + train_scored.columns[:-1].tolist()]
    test_scored = test_scored[["customer_id"] + test_scored.columns[:-1].tolist()]
    validation_scored = validation_scored[["customer_id"] + validation_scored.columns[:-1].tolist()]

    os.makedirs("data/scored", exist_ok=True)
    train_scored.to_pickle(os.path.join("data", "scored", "train.pkl"))
    test_scored.to_pickle(os.path.join("data", "scored", "test.pkl"))
    validation_scored.to_pickle(os.path.join("data", "scored", "validation.pkl"))


if __name__ == "__main__":
    main()
