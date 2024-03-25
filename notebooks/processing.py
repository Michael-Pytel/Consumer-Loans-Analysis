import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


# as previously we'll map it to boolean variable
def create_has_dependents(X):
    return (
        X["HAS_DEPENDENTS"]
        .apply(lambda x: False if x == 0 else True)
        .astype(np.float64)
        .to_frame()
    )


# well keep relation between each education levels
class EducationEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.education_map = [
            "Primary school",
            "Middle school",
            "Highschool",
            "Other",
            "Post secondary school",
            "Vocational school",
            "College",
            "University",
            "Post-graduate",
        ]

    def fit(self, X, y=None):
        return self

    def set_output(self, *args, **kwargs):
        return self

    def transform(self, X):
        return X.map(
            lambda x: self.education_map.index(x) / len(self.education_map)
        ).astype(np.float64)


# using iqr
class RemoveOutliers(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1

        self.r = q3 + 1.5 * iqr
        self.l = q1 - 1.5 * iqr
        return self

    def transform(self, X):
        cols = X.columns
        X_ = X.copy()

        X_ = X_.mask(X_ > self.r, self.r, axis=1)
        X_ = X_.mask(X_ < self.l, self.l, axis=1)
        return pd.DataFrame(X_, columns=cols)

    def set_output(self, *args, **kwargs):
        return self


# chence here there are many missing values we'll try to predict
# them based on all numerical data we'll have so far
class EconomicSectorEncoderImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.estimator = RandomForestClassifier(
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            min_samples_split=10,
            n_estimators=300,
        )
        self.encoder = make_pipeline(OneHotEncoder(sparse_output=False)).set_output(
            transform="pandas"
        )

    def fit(self, X, y=None):
        train_idx = np.where(X["ECONOMIC_SECTOR"] != "Missing")[0]
        X_train = X.select_dtypes(include="number").loc[train_idx, :]
        y_train = X.loc[train_idx, "ECONOMIC_SECTOR"]
        self.estimator.fit(X_train, y_train)
        self.encoder.fit(y_train.to_frame())

        return self

    def transform(self, X):
        X = X.copy()

        pred_idx = np.where(X["ECONOMIC_SECTOR"] == "Missing")[0]
        X_pred = X.select_dtypes(include="number").loc[pred_idx, :]
        X.loc[pred_idx, "ECONOMIC_SECTOR"] = self.estimator.predict(X_pred)

        encoded = self.encoder.transform(X["ECONOMIC_SECTOR"].to_frame())
        X.drop("ECONOMIC_SECTOR", axis=1, inplace=True)

        return pd.concat([X, encoded], axis=1)

    def set_output(self, *args, **kwargs):
        return self


# we'll assume that relation between employee numbers is relevant
# and map it as half of possible numbers where 2000 indicates the highest one
class EmployeeNoEncoderImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.employee_no_map = {
            "> 1.000": 2000,
            "between 0-10": 5,
            "between 101-250": 175,
            "between 11-20": 15,
            "between 21-50": 35,
            "between 251-500": 375,
            "between 501-1.000": 750,
            "between 51-100": 75,
        }

        self.estimator = RandomForestClassifier(
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            min_samples_split=10,
            n_estimators=300,
        )

    def fit(self, X, y=None):
        train_idx = np.where(X["EMPLOYEE_NO"] != "Missing")[0]
        X_train = X.select_dtypes(include="number").loc[train_idx, :]
        y_train = X.loc[train_idx, "EMPLOYEE_NO"]
        self.estimator.fit(X_train, y_train)

        return self

    def transform(self, X):
        X = X.copy()

        pred_idx = np.where(X["EMPLOYEE_NO"] == "Missing")[0]
        X_pred = X.select_dtypes(include="number").loc[pred_idx, :]
        X.loc[pred_idx, "EMPLOYEE_NO"] = self.estimator.predict(X_pred)

        X["EMPLOYEE_NO"] = X["EMPLOYEE_NO"].map(
            lambda x: (self.employee_no_map[x] - 5) / (2000 - 5)
        )

        return X

    def set_output(self, *args, **kwargs):
        return self


class RenameColumn(BaseEstimator, TransformerMixin):
    def __init__(self, old_column_name, new_column_name):
        self.old_column_name = old_column_name
        self.new_column_name = new_column_name

    def transform(self, X):
        X.rename(columns={self.old_column_name: self.new_column_name}, inplace=True)
        return X

    def fit(self, X, y=None):
        return self

    def set_output(self, *args, **kwargs):
        return self


# utility function to keep all column names proper and easily process all data
class ProcessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.fit_transform(X)
            self.reset_columns(X)
        return self

    def transform(self, X):
        for transformer in self.transformers:
            X = transformer.transform(X)
            self.reset_columns(X)
        return X

    @staticmethod
    def reset_columns(X):
        X.columns = [col.split("__")[-1] for col in X.columns]
