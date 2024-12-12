import pandas as pd

from biofefi.options.enums import ProblemTypes


class DataLoaderChecker:
    """
    A utility class for validating data-related conditions.
    """

    def __init__(self, X=None, y=None):
        """
        Initialises the DataLoaderChecker with features and targets.

        Args:
            - X: Features (pd.DataFrame or np.ndarray).
            - y: Targets/Labels (pd.Series, pd.DataFrame, or np.ndarray).
        """
        self.X = X
        self.y = y

    @staticmethod
    def assert_synthetic_data_options(options):
        """
        Validates the synthetic data options based on the problem type.

        Args:
            - options: SyntheticDataOptions object containing all settings.

        Raises:
            - AssertionError: If any validation fails.
        """
        if options.problem_type == ProblemTypes.Classification:
            assert options.n_classes > 0, "Number of classes must be positive."
            assert options.n_features > 0, "Number of features must be positive."
            assert (
                options.n_informative > 0
            ), "Number of informative features must be positive."
            assert (
                options.n_informative <= options.n_features
            ), "Number of informative features cannot exceed the total number of features."

        elif options.problem_type == ProblemTypes.Regression:
            assert (
                options.reg_n_informative > 0
            ), "Number of informative features must be positive."
            assert options.n_features > 0, "Number of features must be positive."
            assert options.reg_n_targets > 0, "Number of targets must be positive."

        else:
            raise ValueError(f"Unsupported problem type: {options.problem_type}")

    def _assert_non_empty_dataframe(self):
        """
        Validates that the DataFrame is not empty.

        Raises:
            - AssertionError: If the DataFrame is empty.
        """
        assert not self.X.empty, "The DataFrame is empty."

    def _assert_non_empty_data(self):
        """
        Validates that the data (X, y) is not empty.

        Raises:
            - AssertionError: If X or y is empty.
        """
        assert self.X is not None and len(self.X) > 0, "The features (X) are empty."
        assert self.y is not None and len(self.y) > 0, "The targets (y) are empty."

    def _assert_no_missing_values(self):
        """
        Validates that the data (X, y) contains no missing values.

        Raises:
            - AssertionError: If X or y contains missing values.
        """
        if isinstance(self.X, pd.DataFrame):
            assert (
                not self.X.isnull().values.any()
            ), "The features (X) contain missing values."
        elif hasattr(self.X, "shape"):  # For numpy arrays
            assert not pd.isnull(
                self.X
            ).any(), "The features (X) contain missing values."

        if isinstance(self.y, (pd.DataFrame, pd.Series)):
            assert (
                not self.y.isnull().values.any()
            ), "The targets (y) contain missing values."
        elif hasattr(self.y, "shape"):  # For numpy arrays
            assert not pd.isnull(
                self.y
            ).any(), "The targets (y) contain missing values."

    def perform_data_checks(self):
        """
        Performs all the data checks:
        1. Ensures data is not empty.
        2. Ensures there are no missing values.

        Raises:
            - AssertionError: If any of the checks fail.
        """
        self._assert_non_empty_data()
        self._assert_no_missing_values()
        if isinstance(self.X, pd.DataFrame):
            self._assert_non_empty_dataframe()
