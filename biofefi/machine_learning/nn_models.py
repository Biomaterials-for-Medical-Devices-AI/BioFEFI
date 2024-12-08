import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from biofefi.machine_learning.nn_networks import BaseNetwork
from biofefi.options.enums import ProblemTypes, ModelNames, OptimiserTypes
from biofefi.options.ml import BrnnOptions
from biofefi.utils.custom_loss import compute_brnn_loss


class BayesianRegularisedNN(
    BaseNetwork, BaseEstimator, ClassifierMixin, RegressorMixin
):
    """
    This class defines a Bayesian Regularized Neural Network
    for regression and classification tasks.

    Attributes:
    - problem_type: The type of problem (Regression or Classification).
    - layer1: The first linear layer of the network.
    - layer2: The second linear layer of the network.
    - output_layer: The output linear layer of the network.
    """

    def __init__(self, problem_type: ProblemTypes):
        super().__init__()
        self._name = ModelNames.BRNN
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
            x (torch.Tensor): The input data to the network.

        Returns:
            torch.Tensor: The output after applying the forward
            pass through the network.

        Raises:
            ValueError: If an unsupported problem type is specified.
        """
        # Apply LeakyReLU activation to hidden layers
        x = F.leaky_relu(self.layer1(x), negative_slope=0.01)
        x = F.leaky_relu(self.layer2(x), negative_slope=0.01)
        x = self.output_layer(x)

        # Apply appropriate activation based on problem type
        if self.problem_type == ProblemTypes.Classification:
            return torch.sigmoid(x) if x.size(1) == 1 else torch.softmax(x, dim=1)

        elif self.problem_type == ProblemTypes.Regression:
            return x
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    def _get_input_output_dims(self, X, y):
        """
        Given the features (X) and target (y), determine the
        input and output dimensions.

        Parameters:
        - X: Features (numpy array or tensor)
        - y: Target values (numpy array or tensor)

        Returns:
        - input_dim: Number of features (columns in X)
        - output_dim: Number of classes (classification) or
        1 (regression)
        """
        # Convert to tensor if not already
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Adjust target `y` for classification tasks
        if self.problem_type == ProblemTypes.Classification:
            y = y.squeeze().long()

        input_dim = X.shape[1]

        # Determine number of output classes
        if self.problem_type == ProblemTypes.Classification:
            output_dim = len(torch.unique(y))  # Number of unique classes
        else:
            output_dim = 1  # Regression has one output

        return input_dim, output_dim

    def fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Bayesian Regularized Neural Network.

        Args:
            X (np.ndarray): The input training data.
            y (np.ndarray): The target training data.

        Returns:
            self: The trained model.
        """
        input_dim, output_dim = self._get_input_output_dims(X, y)
        self._initialize_network(input_dim, output_dim)

        # Set the model to training mode
        self.train()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BrnnOptions.batch_size, shuffle=True
        )

        for epoch in range(BrnnOptions.epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batch_X)

                # Compute total loss
                loss = compute_brnn_loss(self, outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{BrnnOptions.epochs}, Loss: {epoch_loss:.4f}")

        return self

    def predict(self, X, return_probs=False) -> np.ndarray:
        """
        Predict outputs for the given input data.

        Args:
            X: The input data features for prediction (DataFrame, numpy array, or tensor).
            return_probs (bool): Whether to return probabilities instead of class labels.

        Returns:
            np.ndarray: The predicted outputs as a numpy array.
        """
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Convert to tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.eval()

        with torch.no_grad():
            outputs = self(X)
            if self.problem_type == ProblemTypes.Classification:
                if outputs.size(1) == 1:  # Binary classification
                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                    return (
                        probabilities
                        if return_probs
                        else (probabilities > BrnnOptions.classification_cutoff).astype(
                            int
                        )
                    )
                else:  # Multi-class classification
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    return (
                        probabilities
                        if return_probs
                        else np.argmax(probabilities, axis=1)
                    )
            elif self.problem_type == ProblemTypes.Regression:
                return outputs.cpu().numpy()
            else:
                raise ValueError(f"Unsupported problem type: {self.problem_type}")
