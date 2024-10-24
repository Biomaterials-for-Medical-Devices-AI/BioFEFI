import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from biofefi.options.enums import DataSplitMethods, Normalisations


class DataBuilder:
    """
    Data builder class
    """

    _normalization_dict = {
        Normalisations.MinMax: MinMaxScaler,
        Normalisations.Standardization: StandardScaler,
    }

    def __init__(self, opt: argparse.Namespace, logger: object = None) -> None:
        self._path = opt.data_path
        self._data_split = opt.data_split
        self._random_state = opt.random_state
        self._logger = logger
        self._normalization = opt.normalization
        self._numerical_cols = "all"
        self._n_bootstraps = opt.n_bootstraps

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data from a csv file

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The training data (X) and the targets (y)
        """
        df = pd.read_csv(self._path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

    def _generate_data_splits(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Dict[str, List[pd.DataFrame]]:
        """Generate data splits for bootstrapping.

        Args:
            X (pd.DataFrame): The training data.
            y (pd.DataFrame): The prediction targets.

        Raises:
            NotImplementedError: Tried to use an unimplemented data split method.

        Returns:
            Dict[str, List[pd.DataFrame]]: The bootstrapped data.
        """
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        if self._data_split["type"].lower() == DataSplitMethods.Holdout:
            for i in range(self._n_bootstraps):
                self._logger.info(
                    f"Using holdout data split with test size {self._data_split['test_size']} for bootstrap {i+1}"
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self._data_split["test_size"],
                    random_state=self._random_state + i,
                )
                X_train_list.append(X_train)
                X_test_list.append(X_test)
                y_train_list.append(y_train)
                y_test_list.append(y_test)
        else:
            raise NotImplementedError(
                f"Data split type {self._data_split['type']} is not implemented"
            )

        return {
            "X_train": X_train_list,
            "X_test": X_test_list,
            "y_train": y_train_list,
            "y_test": y_test_list,
        }

    def _normalise_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalise data using MinMaxScaler

        Parameters
        ----------
        data : pd.DataFrame
            The data to normalise

        Returns
        -------
        X : pd.DataFrame
            Dataframe of normalised data
        """
        if self._normalization.lower() == Normalisations.NoNormalisation:
            return data

        self._logger.info(f"Normalising data using {self._normalization}...")

        scaler = self._normalization_dict.get(self._normalization.lower())
        if not scaler:
            raise ValueError(
                f"Normalization {self._normalization} is not available. "
                f"Choices are {self._normalization_dict.keys()}"
            )
        scaler = scaler()  # create the scaler object

        if isinstance(self._numerical_cols, str) and self._numerical_cols == "all":
            self._numerical_cols = data.columns
        elif type(self._numerical_cols) == pd.Index:
            pass
        else:
            raise TypeError("numerical_cols must be a list of columns or 'all'.")
        data[self._numerical_cols] = scaler.fit_transform(data[self._numerical_cols])
        return data

    def ingest(self):
        X, y = self._load_data()
        X_norm = self._normalise_data(X)
        data = self._generate_data_splits(X_norm, y)

        return TabularData(
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train=data["y_train"],
            y_test=data["y_test"],
        )


@dataclass
class TabularData:
    # X_train as a list of dataframes
    X_train: list[pd.DataFrame]
    X_test: list[pd.DataFrame]
    y_train: list[pd.DataFrame]
    y_test: list[pd.DataFrame]