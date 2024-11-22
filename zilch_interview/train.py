import os
import pickle as pkl
import yaml
import argparse

from catboost import CatBoostRegressor, Pool

CAT_FEATURES = [
    "occupation",
    "payment_of_min_amount",
    "payment_behaviour"
]


def train(
    data: CatBoostRegressor,
    learning_rate: float,
    iterations: int,
    depth: int,
    l2_leaf_reg: float,
    border_count: int,
    seed: int
) -> CatBoostRegressor:

    X = data.drop(columns=["credit_score_target"])
    y = data["credit_score_target"]
    
    print(X[CAT_FEATURES].info())
    
    train_pool = Pool(X, y, cat_features=CAT_FEATURES)

    # Train the model
    model = CatBoostRegressor(
        learning_rate=learning_rate,
        iterations=iterations,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        random_seed=seed,
        one_hot_max_size=30,
        loss_function="RMSE"
    )

    model.fit(train_pool)

    return model


def main():
    # Get hyperparameters
    params = yaml.safe_load(open("params.yaml"))["train"]
    learning_rate = params["model_params"]["learning_rate"]
    iterations = params["model_params"]["iterations"]
    depth = params["model_params"]["depth"]
    l2_leaf_reg = params["model_params"]["l2_leaf_reg"]
    border_count = params["model_params"]["border_count"]
    seed = params["model_params"]["random_seed"]

    # Read input arguments from the command line
    parser = argparse.ArgumentParser(description="Model training pipeline step")
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    args = parser.parse_args()

    # Load the data
    with open(args.train_data, "rb") as f:
        train_data = pkl.load(f)
        
    train_data.info()

    cb_model = train(
        train_data,
        learning_rate,
        iterations,
        depth,
        l2_leaf_reg,
        border_count,
        seed
    )

    # Save the model
    os.makedirs("models", exist_ok=True)
    with open("models/catboost_model.pkl", "wb") as f:
        pkl.dump(cb_model, f)


if __name__ == "__main__":
    main()
