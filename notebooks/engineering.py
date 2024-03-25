from sklearn.base import BaseEstimator, TransformerMixin


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
            "EDUCATION": 0.4,
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
            + X["EDUCATION"] * self.feature_weights["EDUCATION"]
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
