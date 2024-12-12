from sklearn.datasets import make_classification, make_regression

from biofefi.options.enums import ProblemTypes
from biofefi.options.synthetic_data_opts import SyntheticDataOptions


class CreateSyntheticData:
    """
    This class creates synthetic data for classification
    and regression problems.

    Args:
        - problem_type (ProblemTypes): The type of problem(
            classification or regression).
        - synthetic_options (SyntheticDataOptions): The options
        for creating synthetic data.
        - logger (object): The logger object to log messages.
    """

    def __init__(
        self,
        problem_type: ProblemTypes,
        synthetic_options: SyntheticDataOptions,
        logger: object = None,
    ) -> None:
        """
        Initialize the class with the problem type,
        synthetic data options and logger.
        """
        self._logger = logger
        self._problem_type = problem_type
        self.synthetic_options = synthetic_options
        self.initialise = False
        self.X = None
        self.y = None

    def _create_classification_data(self) -> None:
        """
        Creates synthetic data for classification problems.

        Returns:
            - Tuple: The synthetic data and labels.

        Raises:
            - ValueError: If an error occurs during the data creation.
        """

        if (
            self.synthetic_options.use_synthetic_data
            and self._problem_type == ProblemTypes.Classification
        ):
            self._logger.info(
                "Requested to create synthetic data for Classification..."
            )

            try:
                X, y = make_classification(
                    n_samples=self.synthetic_options.n_samples,
                    n_features=self.synthetic_options.n_features,
                    shuffle=self.synthetic_options.is_shuffle,
                    random_state=self.synthetic_options.random_state,
                    n_informative=self.synthetic_options.n_informative,
                    n_redundant=self.synthetic_options.n_redudant,
                    n_repeated=self.synthetic_options.n_repeated,
                    n_classes=self.synthetic_options.n_classes,
                    n_clusters_per_class=self.synthetic_options.n_clusters_per_class,
                    weights=self.synthetic_options.data_weights,
                    flip_y=self.synthetic_options.flip_y,
                    class_sep=self.synthetic_options.class_sep,
                    scale=self.synthetic_options.data_scale,
                )
                return X, y

            except Exception as e:
                raise ValueError(
                    f"Error creating synthetic data for classification: {e}"
                )

    def _create_regression_data(self) -> None:
        """
        Creates synthetic data for regression problems.

        Returns:
            - Tuple: The synthetic data and labels.

        Raises:
            - ValueError: If an error occurs during the data creation.
        """

        if (
            self.synthetic_options.use_synthetic_data
            and self._problem_type == ProblemTypes.Regression
        ):
            self._logger.info("Requested to create synthetic data for Regression...")

            try:
                X, y = make_regression(
                    n_samples=self.synthetic_options.n_samples,
                    n_features=self.synthetic_options.n_features,
                    shuffle=self.synthetic_options.is_shuffle,
                    random_state=self.synthetic_options.random_state,
                    n_informative=self.synthetic_options.reg_n_informative,
                    n_targets=self.synthetic_options.reg_n_targets,
                    bias=self.synthetic_options.reg_bias,
                    noise=self.synthetic_options.reg_noise,
                    coef=self.synthetic_options.reg_coef,
                )
                return X, y

            except Exception as e:
                raise ValueError(f"Error creating synthetic data for regression: {e}")

    def data_initialisation(self) -> None:
        """
        Initialises the synthetic data creation.

        Raises:
            - InitialisationError: If an error occurs during
            the initialisation.
        """
        if self.synthetic_options.use_synthetic_data:
            self.initialise = True
            self._logger.info("Initialising synthetic data creation...")

            try:
                # Initialise and create synthetic data for classification
                if self._problem_type == ProblemTypes.Classification:
                    self.X, self.y = self._create_classification_data()

                # Initialise and create synthetic data for regression
                elif self._problem_type == ProblemTypes.Regression:
                    self.X, self.y = self._create_regression_data()

                else:
                    raise ValueError(f"Unsupported problem type: {self._problem_type}")

            except Exception as e:
                raise InitialisationError(f"An Error initialising synthetic data: {e}")


class InitialisationError(Exception):
    """
    Custom exception error will be raised if an
    error occurs during the initialisation
    of the synthetic data.
    """

    pass
