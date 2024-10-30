import argparse
from sklearn.datasets import make_classification, make_regression


class SyntheticData:
    """
    This class creates synthetic data for classification
    and regression problems.
    """

    def __init__(self, opt: argparse.Namespace, logger: object = None) -> None:
        """
        Initialize the class

        Parameters
        ----------
        opt : argparse.Namespace
            The options for synthetic data creation.

        logger : object
            The logger object.
        """
        self._logger = logger
        self._opt = opt

    def create_data(self) -> None:
        """
        This method creates synthetic data for classification
        and regression problems.

        Options
        -------
        classifcation:
            Chooses classification problem type.

        regression:
            Chooses regression problem type.

        Returns
        -------
        The synthetic data for classification or regression
        in the form of X and y.

        Raises
        ------
        ValueError
            If there is an error creating synthetic data.

        """

        if self._opt.is_synthetic_data:

            self._logger.info("Requested to create synthetic data...")

            if self._opt.problem_type == "classification":
                self._logger.info("Creating synthetic data for classification...")

                try:

                    X, y = make_classification(
                        n_samples=self._opt.num_samples,
                        n_features=self._opt.num_features,
                        n_informative=self._opt.num_informative,
                        n_redundant=self._opt.num_redundant,
                        n_repeated=self._opt.num_repeated,
                        n_classes=self._opt.num_classes,
                        n_clusters_per_class=self._opt.num_clusters_per_class,
                        weights=self._opt.data_weights,
                        flip_y=self._opt.flip_y,
                        class_sep=self._opt.class_sep,
                        scale=self._opt.data_scale,
                        shuffle=self._opt.is_shuffle,
                        random_state=self._opt.random_state,
                    )

                    return X, y

                except Exception as e:
                    self._logger.error(
                        f"Error creating synthetic data for classification: {e}"
                    )
                    raise ValueError(
                        f"Error creating synthetic data for classification: {e}"
                    )

            elif self._opt.problem_type == "regression":
                self._logger.info("Creating synthetic data for regression...")

                try:

                    X, y = make_regression(
                        n_samples=self._opt.num_samples,
                        n_features=self._opt.num_features,
                        n_informative=self._opt.reg_informative,
                        n_targets=self._opt.num_targets,
                        bias=self._opt.reg_bias,
                        effective_rank=self._opt.reg_effective_rank,
                        noise=self._opt.reg_noise,
                        shuffle=self._opt.is_shuffle,
                        coef=self._opt.reg_coef,
                        random_state=self._opt.random_state,
                    )

                    return X, y

                except Exception as e:
                    self._logger.error(
                        f"Error creating synthetic data for regression: {e}"
                    )
                    raise ValueError(
                        f"Error creating synthetic data for regression: {e}"
                    )

            else:
                self._logger.error(
                    f"Problem type {self._opt.problem_type} is not supported"
                )
                raise ValueError(
                    f"Problem type {self._opt.problem_type} is not supported"
                )

        else:
            self._logger.info("Synthetic data creation is not requested")
            self._loggor.info("Skipping synthetic data creation.")
