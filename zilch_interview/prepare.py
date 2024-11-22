# Define a custom transformer to handle the whole data cleaning process
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

import argparse
import os
import numpy as np
import pandas as pd


DROP_COLS = [
    "customer_id",
    "month",
    "name",
    "address",
    "email",
    "national_insurance_number",
    "num_of_loan",
    "credit_utilization_ratio_2",
    "credit_utilization_ratio_3",
    "credit_score_target"
]

REMOVE_OUTLIER_FEATURES = [
    "age",
    "annual_income",
    "monthly_inhand_salary",
    "num_bank_accounts",
    "num_credit_card",
    "interest_rate",
    "delay_from_due_date",
    "num_of_delayed_payment",
    "changed_credit_limit",
    "num_credit_inquiries",
    "outstanding_debt",
    "credit_utilization_ratio",
    "total_emi_per_month",
    "amount_invested_monthly",
    "monthly_balance"
]

LOAN_TYPES = [
    "Not Specified",
    "Credit-Builder Loan",
    "Personal Loan",
    "Debt Consolidation Loan",
    "Student Loan",
    "Payday Loan",
    "Mortgage Loan",
    "Auto Loan",
    "Home Equity Loan"
]


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            drop_cols: List = None,
            remove_outlier_features: List = None,
            remove_outlier_factor: float = 2.5,
            loan_types: List = None
    ):
        if drop_cols is None:
            self.drop_cols = DROP_COLS
        else:
            self.drop_cols = drop_cols

        if remove_outlier_features is None:
            self.remove_outlier_features = REMOVE_OUTLIER_FEATURES
        else:
            self.remove_outlier_features = remove_outlier_features

        if loan_types is None:
            self.loan_types = LOAN_TYPES
        else:
            self.loan_types = loan_types

        self.remove_outlier_factor = remove_outlier_factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X = X.drop(columns=[col for col in self.drop_cols if col in X.columns])

        # Strip underscores from the age column
        X["age"] = X["age"].apply(lambda x: x.strip("_")).astype(int)

        # Remove blank lines from the occupation column
        X["occupation"] = X["occupation"].apply(lambda x: "Unknown" if x == "_______" else x)

        # Parse the credit_history_age column
        X["credit_history_age"] = X["credit_history_age"].apply(self.parse_credit_history_age)

        # Remove garbage values from the payment_behaviour column
        X["payment_behaviour"] = X["payment_behaviour"].replace("!@9#%8", "Unknown")

        # Type of loan transformer
        X[self.loan_types] = self.loan_type_counter(X[["type_of_loan"]])
        X = X.drop(columns=["type_of_loan"])

        # Remove underscores from changed_credit_limit
        X["changed_credit_limit"] = X["changed_credit_limit"].replace('_', np.nan).astype(float)

        # Remove outliers from the remaining columns
        for feature in self.remove_outlier_features:
            X[feature] = self.remove_outliers(X[feature])

        return X

    def remove_outliers(self, X):
        X = X.copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.remove_outlier_factor * iqr
        upper_bound = q3 + self.remove_outlier_factor * iqr

        X = X.apply(lambda x: x if lower_bound <= x <= upper_bound else None)
        return X

    def loan_type_counter(self, X):
        X = X.copy()
        X = X.fillna("")
        for loan_type in self.loan_types:
            X[loan_type] = X["type_of_loan"].apply(lambda x: x.count(loan_type))
        X = X.drop(columns=["type_of_loan"])
        return X

    @staticmethod
    def parse_credit_history_age(age_string: str) -> int:
        """Parses a string to extract an integer credit history age value.

        Args:
            age_string (str): The string containing the credit history age value.

        Returns:
            int: The parsed integer credit history age value.
        """
        try:
            years = int(age_string.split(" ")[0])
        except (AttributeError, ValueError, IndexError):
            return None

        try:
            months = int(age_string.split(" ")[3])
        except (AttributeError, ValueError, IndexError):
            return 0

        return years * 12 + months


def main():
    # Read input arguments from the command line
    parser = argparse.ArgumentParser(description="Data cleaning pipeline step")
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    parser.add_argument("--test_data", type=str, help="Path to the test data")
    parser.add_argument("--validation_data", type=str, help="Path to the validation data")
    parser.add_argument("--remove_outlier_factor", type=float, help="Factor to remove outliers")
    args = parser.parse_args()

    # Load the data
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)
    validation_data = pd.read_csv(args.validation_data)

    # Initialize the run custom transformer
    data_cleaner = DataCleaner(remove_outlier_factor=args.remove_outlier_factor)
    X_train, y_train = data_cleaner.fit_transform(train_data), train_data["credit_score_target"]
    X_test, y_test = data_cleaner.fit_transform(test_data), test_data["credit_score_target"]
    X_validation = data_cleaner.fit_transform(validation_data)

    # Combine and save the transformed data
    train_tf = pd.concat([X_train, y_train], axis=1)
    test_tf = pd.concat([X_test, y_test], axis=1)

    os.makedirs(os.path.join("data", "cleaned"), exist_ok=True)
    train_tf.to_pickle("data/cleaned/train.pkl")
    test_tf.to_pickle("data/cleaned/test.pkl")
    X_validation.to_pickle("data/cleaned/validation.pkl")


# Set up DVC pipeline step
if __name__ == "__main__":
    main()
