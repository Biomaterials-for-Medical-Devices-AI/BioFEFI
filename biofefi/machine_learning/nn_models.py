import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from biofefi.machine_learning.nn_networks import BaseNetwork
from biofefi.options.enums import ModelNames, OptimiserTypes, ProblemTypes
from biofefi.options.ml import BrnnOptions


class BayesianRegularisedNNClassifier(BaseNetwork, BaseEstimator, ClassifierMixin):
    """
    This class defines a Bayesian Regularized Neural
    Network for classification tasks.

    Args:
        problem_type (ProblemTypes): The type of problem
        (Classification).
        - layer1: The first linear layer of the network.
        - layer2: The second linear layer of the network.
        - output_layer: The output linear layer of the network.
    """

    def __init__(self, problem_type: ProblemTypes):
        """
        Initializes the BayesianRegularisedNNClassifier class.
        """
        super().__init__()
        self._name = ModelNames.BRNNClassifier
        self.problem_type = problem_type
        self.layer1, self.layer2, self.output_layer = None, None, None

    def _initialize_network(self, input_dim, output_dim):
        """
        Initializes the network layers based on the input
        and output dimensions.

        Args:
            input_dim (int): The input dimension of the data.
            output_dim (int): The output dimension of the
            data, determined dynamically.
        """
        # Define hidden layers and output layer
        self.layer1 = nn.Linear(input_dim, BrnnOptions.hidden_dim)
        self.layer2 = nn.Linear(BrnnOptions.hidden_dim, BrnnOptions.hidden_dim)
        self.output_layer = nn.Linear(BrnnOptions.hidden_dim, output_dim)

        # Initialize weights and optimizer
        self._initialise_weights()
        self._get_num_params()
        self._make_optimizer(OptimiserTypes.Adam, BrnnOptions.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output after applying the forward
            pass through the network.

        Raises:
            ValueError: If an error occurs during the forward pass.
        """
        if self.problem_type == ProblemTypes.Classification:
            try:
                x = F.leaky_relu(self.layer1(x), negative_slope=0.01)
                x = F.leaky_relu(self.layer2(x), negative_slope=0.01)
                x = self.output_layer(x)
                return torch.sigmoid(x) if x.size(1) == 1 else torch.softmax(x, dim=1)
            except Exception as e:
                raise ValueError(
                    f"Error occured during forward pass of BRNN Classifier: {e}"
                )

    def fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Bayesian Regularized Neural Network.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.

        Raises:
            ValueError: If an error occurs during training.
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).squeeze().long()
        input_dim = X.shape[1]

        if self.problem_type == ProblemTypes.Classification:
            output_dim = len(torch.unique(y))

        if self.problem_type == ProblemTypes.Classification:
            try:
                self._initialize_network(input_dim, output_dim)
                self.train()
                self.train_brnn(X, y)
            except Exception as e:
                raise ValueError(
                    f"Error occured during fitting of BRNN Classifier: {e}"
                )

    def predict(self, X, return_probs=False) -> np.ndarray:
        """
        Predict the target values using the trained BRNN Regressor.

        Args:
            X (np.ndarray): The input data.
            return_probs (bool): Whether to return the predicted

        Returns:
            np.ndarray: The predicted target values.

        Raises:
            ValueError: If an error occurs during prediction.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.problem_type == ProblemTypes.Classification:
            try:
                self.eval()
                with torch.no_grad():
                    outputs = self(X)

                    if outputs.size(1) == 1:  # Binary classification
                        probabilities = torch.sigmoid(outputs).cpu().numpy()
                        return (
                            probabilities
                            if return_probs
                            else (
                                probabilities > BrnnOptions.classification_cutoff
                            ).astype(int)
                        )

                    else:  # Multi-class classification
                        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                        return (
                            probabilities
                            if return_probs
                            else np.argmax(probabilities, axis=1)
                        )

            except Exception as e:
                raise ValueError(
                    f"Error occured during prediction of BRNN Classifier: {e}"
                )


class BayesianRegularisedNNRegressor(BaseNetwork, BaseEstimator, RegressorMixin):
    """
    This class defines a Bayesian Regularized Neural
    Network for regression tasks.

    Args:
        problem_type (ProblemTypes): The type of problem
        (Regression).
        - layer1: The first linear layer of the network.
        - layer2: The second linear layer of the network.
        - output_layer: The output linear layer of the network.
    """

    def __init__(self, problem_type: ProblemTypes):
        """
        Initializes the BayesianRegularisedNNRegressor class.
        """
        super().__init__()
        self.problem_type = problem_type
        self._name = ModelNames.BRNNRegressor
        self.layer1, self.layer2, self.output_layer = None, None, None

    def _initialize_network(self, input_dim, output_dim):
        """
        Initializes the network layers for BRNN regression.

        Args:
            input_dim (int): The input dimension of the data.
            output_dim (int): The output dimension of the
            data, determined dynamically.
        """
        # Define hidden layers and output layer
        self.layer1 = nn.Linear(input_dim, BrnnOptions.hidden_dim)
        self.layer2 = nn.Linear(BrnnOptions.hidden_dim, BrnnOptions.hidden_dim)
        self.output_layer = nn.Linear(BrnnOptions.hidden_dim, output_dim)

        # Initialize weights and optimizer
        self._initialise_weights()
        self._get_num_params()
        self._make_optimizer(OptimiserTypes.Adam, BrnnOptions.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output after applying the forward
            pass through the network.

        Raises:
            ValueError: If an error occurs during the forward pass.
        """
        if self.problem_type == ProblemTypes.Regression:
            try:
                x = F.leaky_relu(self.layer1(x), negative_slope=0.01)
                x = F.leaky_relu(self.layer2(x), negative_slope=0.01)
                x = self.output_layer(x)
                return x
            except Exception as e:
                raise ValueError(
                    f"Error occured during forward pass of BRNN Regressor: {e}"
                )

    def fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Bayesian Regularized Neural Network.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.

        Raises:
            ValueError: If an error occurs during training.
        """

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).squeeze().long()
        input_dim = X.shape[1]

        if self.problem_type == ProblemTypes.Regression:
            output_dim = 1

        if self.problem_type == ProblemTypes.Regression:
            try:
                self._initialize_network(input_dim, output_dim)
                self.train()
                self.train_brnn(X, y)
            except Exception as e:
                raise ValueError(f"Error occured during fitting of BRNN Regressor: {e}")

    def predict(self, X) -> np.ndarray:
        """
        Predict the target values using the trained BRNN Regressor.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted target values.

        Raises:
            ValueError: If an error occurs during prediction.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.problem_type == ProblemTypes.Regression:
            try:
                self.eval()
                with torch.no_grad():
                    outputs = self(X)
                    return outputs.cpu().numpy()
            except Exception as e:
                raise ValueError(
                    f"Error occured during prediction of BRNN Regressor: {e}"
                )
