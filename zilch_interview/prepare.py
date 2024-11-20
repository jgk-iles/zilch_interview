# Define a custom transformer to handle the whole data cleaning process
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            drop_cols: List = [
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
            ],
            remove_outlier_features: List = [
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
            ],
            remove_outlier_factor: float = 2.5,
            loan_types: List = [
                "Not Specified",
                "Credit-Builder Loan",
                "Personal Loan",
                "Debt Consolidation Loan",
                "Student Loan",
                "Payday Loan",
                "Mortgage Loan",
                "Auto Loan",
                "Home Equity Loan"
            ]):
        self.remove_outlier_features = remove_outlier_features
        self.remove_outlier_factor = remove_outlier_factor
        self.drop_cols = drop_cols
        self.loan_types = loan_types

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.drop_cols)

        # Strip underscores from the age column
        X["age"] = X["age"].apply(lambda x: x.strip("_")).astype(int)

        # Remove blank lines from the occupation column
        X["occupation"] = X["occupation"].apply(lambda x: "Unknown" if x == "_______" else x)

        # Parse the credit_history_age column
        X["credit_history_age"] = X["credit_history_age"].apply(self.parse_credit_history_age)

        # Remove garbage values from the payment_behaviour column
        garbage_remover = SimpleImputer(missing_values="!@9#%8", strategy="constant", fill_value=None)
        X["payment_behaviour"] = garbage_remover.fit_transform(X[["payment_behaviour"]]).flatten()

        # Type of loan transformer
        X[self.loan_types] = self.loan_type_counter(X[["type_of_loan"]])
        X = X.drop(columns=["type_of_loan"])

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
        print(X)
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