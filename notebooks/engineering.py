from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class CreditScorePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self.weights = None

    def fit(self, df, target_column):
        def get_total_score(weights):
            X = df[self.features].values
            return np.dot(X, weights)

        def get_total_score_corr(weights):
            total_score = get_total_score(weights)
            correlation = np.corrcoef(total_score, df[target_column])[0, 1]
            return correlation

        def get_total_score_corr_metric(weights):
            return -np.abs(get_total_score_corr(weights))

        initial_weights = np.ones(len(self.features))
        result = minimize(get_total_score_corr_metric, initial_weights, method='Nelder-Mead')
        self.weights = result.x

    def transform(self, df):
        def get_total_score(weights):
            X = df[self.features].values
            return np.dot(X, weights)

        df["Total_Score"] = get_total_score(self.weights)
        min_score = df['Total_Score'].min()
        max_score = df['Total_Score'].max()
        df['CREDIT_SCORE'] = (df['Total_Score'] - min_score) / (max_score - min_score)
        df.drop(columns=['Total_Score'], inplace=True)
        return df
    
    def set_output(self, *args, **kwargs):
        return self


class CreateHasCurrentAccountColumn(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

    def set_output(self, *args, **kwargs):
        return self


class CalculateCreditScore(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_weights = {
            "INCOME": 0.4,
            "WORK_SENIORITY": 0.3,
            "BUSINESS_AGE": 0.1,
            "LENGTH_RELATIONSHIP_WITH_CLIENT": 0.4,
            "AGE": 0.4,
            "HAS_DEPENDENTS": 0.2,
            "MARITAL_STATUS_married": 0.3,
            "MARITAL_STATUS_single": -0.2,
            "RESIDENTIAL_PLACE_Owner without mortgage": 0.1,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Calculate total score for each individual
        X["Total_Score"] = (
            X["INCOME"] * self.feature_weights["INCOME"]
            + X["WORK_SENIORITY"] * self.feature_weights["WORK_SENIORITY"]
            + X["BUSINESS_AGE"] * self.feature_weights["BUSINESS_AGE"]
            + X["LENGTH_RELATIONSHIP_WITH_CLIENT"]
            * self.feature_weights["LENGTH_RELATIONSHIP_WITH_CLIENT"]
            + X["AGE"] * self.feature_weights["AGE"]

            + X["MARITAL_STATUS_married"]
            * self.feature_weights["MARITAL_STATUS_married"]
            + X["MARITAL_STATUS_single"] * self.feature_weights["MARITAL_STATUS_single"]
            + X["RESIDENTIAL_PLACE_Owner without mortgage"]
            * self.feature_weights["RESIDENTIAL_PLACE_Owner without mortgage"]
        )

        # Normalize total scores to a range between 0-1
        min_score = X["Total_Score"].min()
        max_score = X["Total_Score"].max()
        X["CREDIT_SCORE"] = (X["Total_Score"] - min_score) / (max_score - min_score)

        # Drop the intermediate total score column
        X.drop(columns=["Total_Score"], inplace=True)

        return X

    def set_output(self, *args, **kwargs):
        return self
# class CalculateCreditScore(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.features = ['INCOME', 'WORK_SENIORITY', 'BUSINESS_AGE', 'LENGTH_RELATIONSHIP_WITH_CLIENT', 'AGE',
#                          'RESIDENTIAL_PLACE_Living with family', 'MARITAL_STATUS_married',
#                          'MARITAL_STATUS_single', 'RESIDENTIAL_PLACE_Owner without mortgage', 'PRODUCT_C',
#                          'SALARY_ACCOUNT']
#         self.target_column = 'FINALIZED_LOAN'
#         self.weights = None

#     def fit(self, df):
#         def get_total_score(weights):
#             X = df[self.features].values
#             return np.dot(X, weights)

#         def get_total_score_corr(weights):
#             total_score = get_total_score(weights)
#             correlation = np.corrcoef(total_score, df[self.target_column])[0, 1]
#             return correlation

#         def get_total_score_corr_metric(weights):
#             return -np.abs(get_total_score_corr(weights))

#         initial_weights = np.ones(len(self.features))
#         result = minimize(get_total_score_corr_metric, initial_weights, method='Nelder-Mead')
#         self.weights = result.x
#         return self

#     def transform(self, df):
#         def get_total_score(weights):
#             X = df[self.features].values
#             return np.dot(X, weights)

#         df["Total_Score"] = get_total_score(self.weights)
#         min_score = df['Total_Score'].min()
#         max_score = df['Total_Score'].max()
#         df['CREDIT_SCORE'] = (df['Total_Score'] - min_score) / (max_score - min_score)
#         df.drop(columns=['Total_Score'], inplace=True)
#         return df

#     def fit_transform(self, df):
#         return self.fit(df).transform(df)

#     def set_output(self, *args, **kwargs):
#         return self

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
    def __init__(self, transformers, pipeline):
        from processing import ProcessingTransformer

        self.root_transformer = pipeline
        self.transformers = transformers

    def fit(self, X, y=None):
        X = self.root_transformer.transform(X)

        for transformer in self.transformers:
            X = transformer.fit_transform(X)
            self.reset_columns(X)
        return self

    def transform(self, X):
        X = self.root_transformer.transform(X)

        for transformer in self.transformers:
            X = transformer.transform(X)
            self.reset_columns(X)
        return X

    @staticmethod
    def reset_columns(X):
        X.columns = [col.split("__")[-1] for col in X.columns]
