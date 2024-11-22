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

    X = data.drop(columns=["credit_score_target"])
    y = data["credit_score_target"]

    # Evaluate the model
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


def main():
    EVAL_DIR = "evaluation"
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Get arguments from the command line
    parser = argparse.ArgumentParser(description="Model evaluation pipeline step")
    parser.add_argument("--model", type=str, help="Path to the trained model")
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    parser.add_argument("--test_data", type=str, help="Path to the test data")
    args = parser.parse_args()

    # Load the model and data
    with open(args.model, "rb") as model_file:
        model = pkl.load(model_file)

    train_data = pd.read_pickle(args.train_data)
    test_data = pd.read_pickle(args.test_data)

    with Live(EVAL_DIR) as live:
        evaluate(model, train_data, "train", live)
        evaluate(model, test_data, "test", live)


if __name__ == "__main__":
    main()
