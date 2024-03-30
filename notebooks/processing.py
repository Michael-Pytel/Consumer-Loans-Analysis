import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score


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
        super().__init__()
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


class RenameColumn(BaseEstimator, TransformerMixin):
    def __init__(self, old_column_name, new_column_name):
        super().__init__()
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
        super().__init__()
        self.transformers = transformers

    def fit(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.fit_transform(X)
            ProcessingTransformer.reset_columns(X)
        return self

    def transform(self, X):
        for transformer in self.transformers:
            X = transformer.transform(X)
            ProcessingTransformer.reset_columns(X)
        return X

    def set_output(self, *args, **kwargs):
        return self

    @staticmethod
    def reset_columns(X):
        X.columns = [col.split("__")[-1] for col in X.columns]


class EmployeePartialTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.employee_no_map = {
            "between 0-10": 0,
            "between 11-20": 1,
            "between 21-50": 2,
            "between 51-100": 3,
            "between 101-250": 4,
            "between 251-500": 5,
            "between 501-1.000": 6,
            "> 1.000": 7,
            "Missing": None,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["EMPLOYEE_NO_NUM"] = X["EMPLOYEE_NO"].map(self.employee_no_map)
        X.drop(columns=["EMPLOYEE_NO"], inplace=True)
        return X

    def set_output(self, *args, **kwargs):
        return self


class MyImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self, estimator, target, encoder=None, missing_val="Missing", train=False
    ):
        super().__init__()
        self.estimator = estimator
        self.target = target
        self.encoder = encoder
        self.missing_val = missing_val
        self.train = train
        self.target_cols = []

    def fit(self, X, y=None):
        idx = np.where(X[self.target] != self.missing_val)[0]

        if self.encoder is not None:
            X = self.encoder.fit_transform(X.loc[idx, :])
            ProcessingTransformer.reset_columns(X)
        self.target_cols = MyImputer.get_target_cols(X, self.target)

        if self.train:
            X_train = X.loc[:, ~X.columns.isin(self.target_cols)]
            y_train = X.loc[:, self.target_cols]
            self.estimator.fit(X_train, y_train)

        return self

    def transform(self, X):
        idx = np.where(X[self.target] == self.missing_val)[0]

        if self.encoder is not None:
            X = self.encoder.transform(X)
            ProcessingTransformer.reset_columns(X)

        if idx.any():
            X_pred = X.iloc[idx, ~X.columns.isin(self.target_cols)]
            X.iloc[idx, [list(X.columns).index(col) for col in self.target_cols]] = (
                self.estimator.predict(X_pred)
            )
        return X

    def set_output(self, *args, **kwargs):
        return self

    @staticmethod
    def get_target_cols(X, target):
        return [col for col in X.columns if col.startswith(target) and col != target]


def evaluate_estimator(
    df,
    df_valid,
    target,
    estimator,
    encoder=None,
    missing_val="Missing",
    drop_columns=[],
):
    imputer = ColumnTransformer(
        [
            (
                "imputer",
                MyImputer(
                    estimator=estimator,
                    encoder=encoder,
                    target=target,
                    missing_val=missing_val,
                    train=True,
                ),
                [col for col in df.columns if col not in drop_columns],
            )
        ],
        remainder="passthrough",
    ).set_output(transform="pandas")
    df = df.copy()
    df_valid = df_valid.copy()

    df = imputer.fit_transform(df)
    ProcessingTransformer.reset_columns(df)
    target_cols = MyImputer.get_target_cols(df, target)

    score = None
    idx = np.where(df_valid[target] != missing_val)[0]
    if len(idx) > 0:
        df_valid_test = imputer.transform(df_valid.loc[idx, :])
        ProcessingTransformer.reset_columns(df_valid_test)
        y_test = df_valid_test.loc[idx, target_cols]

        df_valid.loc[idx, target] = missing_val
        df_valid_pred = imputer.transform(df_valid.loc[idx, :])
        ProcessingTransformer.reset_columns(df_valid_pred)
        y_pred = df_valid_pred.loc[idx, target_cols]

        score = f1_score(y_test, y_pred, average="micro")
    return score, imputer


def create_best_estimator(
    study,
    df,
    df_valid,
    target,
    estimator_class,
    encoder=None,
    missing_val="Missing",
    drop_columns=[],
):
    trial = study.best_trial

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    estimator = estimator_class(**trial.params)
    score, pipeline = evaluate_estimator(
        estimator=estimator,
        encoder=encoder,
        df=df,
        df_valid=df_valid,
        target=target,
        drop_columns=drop_columns,
        missing_val=missing_val,
    )

    print(f"Refitted best model f1-score: {score}")
    return pipeline
