import numpy as np
import pandas as pd
from processing import ProcessingTransformer
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CreateHasCurrentAccountColumn(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)

    def create(self, X):
        # Create a copy of the 'CURRENT_ACCOUNT' column with a new name
        X["HAS_CURRENT_ACCOUNT"] = X["CURRENT_ACCOUNT"].astype("object")

        # Replace boolean values with strings in the new column
        X.loc[
            (X["DEBIT_CARD"] == True) & (X["CURRENT_ACCOUNT"] == True),
            "HAS_CURRENT_ACCOUNT",
        ] = "with debit card"
        X.loc[
            (X["DEBIT_CARD"] == False) & (X["CURRENT_ACCOUNT"] == False),
            "HAS_CURRENT_ACCOUNT",
        ] = "no"

        # Replace False values with 'without debit card' only if 'DEBIT_CARD' is False in the new column
        X.loc[
            (X["DEBIT_CARD"] == False) & (X["CURRENT_ACCOUNT"] == True),
            "HAS_CURRENT_ACCOUNT",
        ] = "without debit card"

        X.drop(columns=["DEBIT_CARD", "CURRENT_ACCOUNT"], inplace=True)

        return X

    def fit(self, X, y=None):
        X = self.create(X.copy())
        self.encoder.fit(X["HAS_CURRENT_ACCOUNT"].to_frame())
        return self

    def transform(self, X):
        X = self.create(X)
        encoded = self.encoder.transform(X["HAS_CURRENT_ACCOUNT"].to_frame())
        encoded_df = pd.DataFrame(
            encoded, columns=self.encoder.get_feature_names_out(["HAS_CURRENT_ACCOUNT"])
        )

        X.drop("HAS_CURRENT_ACCOUNT", axis=1, inplace=True)
        X = pd.concat([X, encoded_df], axis=1)
        return X

    def set_output(self, *args, **kwargs):
        return self


class CreateAdditionalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Calculate additional features
        X["LENGTH_RELATIONSHIP_WITH_CLIENT_TO_WORK_SENIORITY"] = (
            X["LENGTH_RELATIONSHIP_WITH_CLIENT"] / X["WORK_SENIORITY"]
        )
        X["INCOME_TO_WORK_SENIORITY_RATIO"] = X["INCOME"] / X["WORK_SENIORITY"]
        X["BUSINESS_AGE_TO_AGE_RATIO"] = X["BUSINESS_AGE"] / X["WORK_SENIORITY"]
        X["LENGTH_RELATIONSHIP_WITH_CLIENT_TO_BUSINESS_AGE"] = (
            X["LENGTH_RELATIONSHIP_WITH_CLIENT"] / X["BUSINESS_AGE"]
        )
        X["INCOME_TO_LENGTH_RELATIONSHIP_WITH_CLIENT"] = (
            X["INCOME"] / X["LENGTH_RELATIONSHIP_WITH_CLIENT"]
        )

        return X

    def set_output(self, *args, **kwargs):
        return self


class EngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers, root_transformer):
        self.transformers = transformers
        self.root_transformer = root_transformer

    def fit(self, X, y=None):
        X = self.root_transformer.transform(X)

        for transformer in self.transformers:
            X = transformer.fit_transform(X)
            ProcessingTransformer.reset_columns(X)
        return self

    def transform(self, X):
        X = self.root_transformer.transform(X)

        for transformer in self.transformers:
            X = transformer.transform(X)
            ProcessingTransformer.reset_columns(X)
        return X

    def set_output(self, *args, **kwargs):
        return self


class FeatureCorrelationEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_combine, target_col, new_name=None, drop=False):
        super().__init__()
        self.cols_to_combine = cols_to_combine
        self.target_col = target_col
        self.weights = np.ones(len(cols_to_combine))
        self.new_name = new_name if new_name is not None else "_".join(cols_to_combine)
        self.drop = drop
        self.scaler = StandardScaler()

    def get_combined_value(self, X):
        return X[self.cols_to_combine].values.dot(self.weights.reshape(-1, 1))

    def get_corr(self, X):
        return np.corrcoef(self.get_combined_value(X).ravel(), X[self.target_col])[0, 1]

    def fit(self, X, y=None):
        def get_score(weights):
            self.weights = weights
            return -np.abs(self.get_corr(X))

        self.weights = minimize(get_score, self.weights, method="Nelder-Mead")["x"]

        new_col = self.get_combined_value(X)
        self.scaler.fit(new_col)

        return self

    def transform(self, X):
        new_col = self.get_combined_value(X)
        new_col = self.scaler.transform(new_col)
        X[self.new_name] = new_col

        if self.drop:
            X.drop(columns=self.cols_to_combine, inplace=True)
        return X

    def set_output(self, *args, **kwargs):
        return self
