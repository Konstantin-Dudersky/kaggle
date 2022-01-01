from typing import Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import pandas as pd
import numpy as np


def regression_report(df, model_name, y_train, y_train_pred, y_test, y_test_pred, sort_by='R2'):
    _df = df.copy()

    if len(_df) == 0:
        tuples = [
            ('R2', 'train'), ('R2', 'test'),
            ('MAE', 'train'), ('MAE', 'test'),
            ('MSE', 'train'), ('MSE', 'test'),
            ('MAPE', 'train'), ('MAPE', 'test'),
        ]
        index = pd.MultiIndex.from_tuples(tuples)
        _df = pd.DataFrame(index=index).T

    _df.loc[model_name] = {
        ('R2', 'train'): r2_score(y_train, y_train_pred),
        ('R2', 'test'): r2_score(y_test, y_test_pred),
        ('MAE', 'train'): mean_absolute_error(y_train, y_train_pred),
        ('MAE', 'test'): mean_absolute_error(y_test, y_test_pred),
        ('MSE', 'train'): mean_squared_error(y_train, y_train_pred),
        ('MSE', 'test'): mean_squared_error(y_test, y_test_pred),
        ('MAPE', 'train'): mean_absolute_percentage_error(y_train, y_train_pred),
        ('MAPE', 'test'): mean_absolute_percentage_error(y_test, y_test_pred),
    }

    _df = _df.sort_values(by=(sort_by, 'test'))

    display(_df)

    return _df


class DisplayDfInPipe(BaseEstimator, TransformerMixin):
    def __init__(self, n: int = 10):
        self.n = n

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> 'DisplayDfInPipe':
        return self

    def transform(self, x: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        display(x.head(self.n))

        return x


class OneHotEncoderMy(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Union[list, str]) -> None:
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

        self.encoders = {}
        for col in self.columns:
            self.encoders[col] = OneHotEncoder(sparse=False, dtype=np.uint8, handle_unknown='ignore')

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> 'OneHotEncoderMy':
        for col in self.columns:
            self.encoders[col].fit(x[col].to_numpy().reshape(-1, 1))
        return self

    def transform(self, x: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        df = x.copy()
        for col in self.columns:
            feature_names = [f"{col}{x[2:]}" for x in self.encoders[col].get_feature_names_out()]

            x_transformed = self.encoders[col].transform(df[col].to_numpy().reshape(-1, 1))
            x_transformed_df = pd.DataFrame(
                data=x_transformed,
                columns=feature_names,
                index=df.index
            )
            df = pd.concat([df, x_transformed_df], axis=1).drop(columns=[col])

        return df


class OrdinalEncoderMy(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        self.encoder = OrdinalEncoder(dtype=np.float64)
        self.columns = columns

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> 'OrdinalEncoderMy':
        self.encoder.fit(x[self.columns])
        return self

    def transform(self, x: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        df = x.copy()

        df[self.columns] = pd.DataFrame(
            data=self.encoder.transform(df[self.columns]),
            columns=df[self.columns].columns,
            index=df.index
        )
        return df


class StandardScalerMy(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Union[list, str]):
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

        self.scaler = StandardScaler()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> 'StandardScalerMy':
        self.scaler.fit(x[self.columns])
        return self

    def transform(self, x: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        df = x.copy()

        df[self.columns] = self.scaler.transform(df[self.columns])
        return df


class SimpleImputerMy(BaseEstimator, TransformerMixin):
    def __init__(
        self, columns: Union[list, str], strategy: str = 'mean', fill_value=None, missing_indicator=False
    ) -> None:
        if type(columns) == list:
            self.columns = columns
        elif type(columns) == str:
            self.columns = [columns]
        else:
            raise TypeError('Wrong type of parameter "columns"')

        self.missing_indicator = missing_indicator
        self.fill_value = fill_value
        self.strategy = strategy
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=self.strategy, fill_value=self.fill_value)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> 'SimpleImputerMy':
        self.imputer.fit(x[self.columns])
        return self

    def transform(self, x: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        df = x.copy()

        df[self.columns] = self.imputer.transform(df[self.columns])

        # добавляем признак с инфо о пропущенных значениях
        for col in self.columns:
            if self.missing_indicator:
                df[f'{col}_isna'] = df[col].isna()

        return df
