# Define a custom transformer to handle the whole data cleaning process
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class StringStripper(BaseEstimator, TransformerMixin):
    def __init__(self, strip_char: str = "_"):
        """Initializes the transformer with the character to strip from the string.

        Args:
            strip_char (str, optional): Character to strip. Defaults to "_".
        """
        self.strip_char = strip_char

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def parse_string(string: str) -> int:
            """Parses a string to extract an integer age value.

            string (str): The string containing the age value. It may contain underscores.

            int: The parsed integer age value. If the string cannot be directly converted to an integer,
                    it attempts to strip underscores and convert again.

            Raises:
                ValueError: If the string cannot be converted to an integer even after stripping underscores.
            """
            try:
                return int(string)
            except ValueError:
                return int(string.strip(self.strip_char))

        X = X.copy()
        X = X.apply(parse_string)
        return X


# class OutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, factor=2.5):
#         self.factor = factor
#         self.lower_bound = None
#         self.upper_bound = None

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()
#         for column in X.columns:
#             q1 = X[column].quantile(0.25)
#             q3 = X[column].quantile(0.75)
#             iqr = q3 - q1

#             self.lower_bound = q1 - self.factor * iqr
#             self.upper_bound = q3 + self.factor * iqr

#             X[column] = X[column].apply(lambda x: x if self.lower_bound <= x <= self.upper_bound else None)
#         return X


class CreditHistoryAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.apply(self.parse_credit_history_age)
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


class LoanTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, loan_types=None):
        if loan_types is None:
            loan_types = [
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
        self.loan_types = loan_types

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.fillna("")
        for loan_type in self.loan_types:
            X[loan_type] = X["type_of_loan"].apply(lambda x: x.count(loan_type))
        X = X.iloc[:, 1:]
        return X


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
        string_stripper = StringStripper()
        X["age"] = string_stripper.fit_transform(X["age"])

        # Remove blank lines from the occupation column
        blank_line_remover = SimpleImputer(missing_values="_______", strategy="constant", fill_value="Unknown")
        X["occupation"] = blank_line_remover.fit_transform(X[["occupation"]]).flatten()

        # Remove -100 values from the num_of_loan column
        minus_100_remover = SimpleImputer(missing_values=-100, strategy="constant", fill_value=None)
        X["num_of_loan"] = minus_100_remover.fit_transform(X[["num_of_loan"]])

        # Parse the credit_history_age column
        credit_history_age_transformer = CreditHistoryAgeTransformer()
        X["credit_history_age"] = credit_history_age_transformer.fit_transform(X["credit_history_age"])

        # Remove garbage values from the payment_behaviour column
        garbage_remover = SimpleImputer(missing_values="!@9#%8", strategy="constant", fill_value=None)
        X["payment_behaviour"] = garbage_remover.fit_transform(X[["payment_behaviour"]]).flatten()

        # Type of loan transformer
        loan_type_transformer = LoanTypeTransformer()
        X[self.loan_types] = loan_type_transformer.fit_transform(X[["type_of_loan"]])
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


# class OutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, factor=2.5):
#         self.factor = factor
#         self.lower_bound = None
#         self.upper_bound = None

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()
#         for column in X.columns:
#             q1 = X[column].quantile(0.25)
#             q3 = X[column].quantile(0.75)
#             iqr = q3 - q1

#             self.lower_bound = q1 - self.factor * iqr
#             self.upper_bound = q3 + self.factor * iqr

#             X[column] = X[column].apply(lambda x: x if self.lower_bound <= x <= self.upper_bound else None)
#         return X
