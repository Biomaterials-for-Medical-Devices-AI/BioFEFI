import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from biofefi.machine_learning.networks import BaseNetwork
import numpy as np


class BayesianRegularisedNeuralNets(
    BaseNetwork, BaseEstimator, ClassifierMixin, RegressorMixin
):
    """
    This class implements Bayesian Regularised Neural Networks compatible with scikit-learn.
    """

    def __init__(self, opt):
        """
        Initializes the BayesianRegularisedNeuralNets class

        Parameters
        ----------
        opt : argparse.Namespace
            The options for the BayesianRegularisedNeuralNets class.
        """
        super().__init__(opt)
        self._name = "BayesianRegularisedNeuralNets"
        self._opt = opt
        self._hidden_dim = opt.hidden_dim

        # Set problem type to classification or regression
        self.problem_type = opt.problem_type

        # Layers will be initialized in `_initialize_network`
        self.layer1 = None
        self.layer2 = None
        self.output_layer = None

    def _initialize_network(self, input_dim, output_dim):
        """
        Initialize the network layers based on the problem type.

        Parameters
        ----------
        input_dim : int
            The input dimension of the data.
        output_dim : int
            The output dimension of the data, determined dynamically.
        """
        # Define hidden layers and output layer
        self.layer1 = nn.Linear(input_dim, self._hidden_dim)
        self.layer2 = nn.Linear(self._hidden_dim, self._hidden_dim)
        self.output_layer = nn.Linear(self._hidden_dim, output_dim)

        # Initialize weights and create optimizer here after layers are defined
        self._initialise_weights()
        self._make_optimizer(self._opt.optimizer_type, self._opt.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input data.
        """
        # Apply LeakyReLU activation to hidden layers
        x = F.leaky_relu(self.layer1(x), negative_slope=0.01)
        x = F.leaky_relu(self.layer2(x), negative_slope=0.01)
        x = self.output_layer(x)

        # Apply appropriate activation based on problem type
        if self.problem_type.lower() == "classification":

            # Use softmax for multi-class classification, sigmoid for binary
            return torch.sigmoid(x) if x.size(1) == 1 else torch.softmax(x, dim=1)

        # No activation for regression
        elif self.problem_type.lower() == "regression":
            return x
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    def bayesian_regularization_loss(
        self, prior_mu: float = None, prior_sigma: float = None
    ) -> torch.Tensor:
        """
        Compute the Bayesian Regularization loss.

        Parameters
        ----------
        prior_mu : float
            The prior mean.

        prior_sigma : float
            The prior standard deviation.
        """
        prior_mu = prior_mu if prior_mu is not None else self._opt.prior_mu
        prior_sigma = prior_sigma if prior_sigma is not None else self._opt.prior_sigma

        if prior_mu is None or prior_sigma is None:
            raise ValueError("Prior mean and standard deviation must be provided")

        # Calculate regularization loss
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.sum((param - prior_mu) ** 2) / (2 * prior_sigma**2)
        return reg_loss

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss based on the problem type and return the total loss including the regularization loss.

        Parameters
        ----------
        outputs : torch.Tensor
            The predicted outputs.

        targets : torch.Tensor
            The target outputs.
        """
        # Determine the predictive loss based on problem type
        if self.problem_type.lower() == "classification":
            if outputs.size(1) == 1:
                predictive_loss = nn.BCELoss()(outputs, targets)
                print("Using BCELoss for binary classification.")
            else:
                predictive_loss = nn.CrossEntropyLoss()(outputs, targets)
                print("Using CrossEntropyLoss for multi-class classification.")
        elif self.problem_type.lower() == "regression":
            predictive_loss = nn.MSELoss()(outputs, targets)
            print("Using MSELoss for regression.")
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

        # Compute regularization loss and add it to predictive loss
        reg_loss = self.bayesian_regularization_loss()
        total_loss = predictive_loss + self._opt.lambda_reg * reg_loss
        return total_loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """
        Trains the Bayesian Regularized Neural Network.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        y : np.ndarray
            The target data.

        validation_data : Optional[Tuple[np.ndarray, np.ndarray]]
            The validation data.
        """
        # Convert input data to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Adjust target `y` for classification tasks
        if self.problem_type.lower() == "classification":

            # Ensure y is 1D and of type Long for CrossEntropyLoss
            y = y.squeeze().long()

        # Convert validation data if provided
        if validation_data is not None:
            val_X, val_y = validation_data
            val_X = torch.tensor(val_X, dtype=torch.float32)
            val_y = torch.tensor(val_y, dtype=torch.float32)

            # Ensure validation y is also 1D for classification
            if self.problem_type.lower() == "classification":
                val_y = val_y.squeeze().long()
            validation_data = (val_X, val_y)

        input_dim = X.shape[1]

        # Determine number of classes if classification
        if self.problem_type.lower() == "classification":

            # Set output_dim dynamically based on unique classes
            self._output_dim = len(torch.unique(y))
        else:
            # Regression output_dim is 1
            self._output_dim = 1

        # Initialize network layers and optimizer
        self._initialize_network(input_dim, self._output_dim)

        # Training loop
        self.train()
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self._opt.batch_size, shuffle=True
        )

        for epoch in range(self._opt.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.compute_loss(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Validation step if validation data is provided
            if validation_data is not None:
                val_X, val_y = validation_data
                val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                self.eval()
                with torch.no_grad():
                    val_outputs = self(val_X)
                    val_loss = self.compute_loss(val_outputs, val_y)
                self.train()
                print(
                    f"Epoch {epoch + 1}/{self._opt.epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss.item():.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{self._opt.epochs}, Loss: {epoch_loss:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for the given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data features.

        Returns
        -------
        np.ndarray
            Predicted outputs as a numpy array.
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            if self.problem_type == "classification":
                if outputs.size(1) == 1:
                    return (
                        (outputs > self._opt.classification_cutoff)
                        .float()
                        .cpu()
                        .numpy()
                    )
                else:
                    return torch.argmax(outputs, dim=1).cpu().numpy()
            elif self.problem_type == "regression":
                return outputs.cpu().numpy()
            else:
                raise ValueError(f"Unsupported problem type: {self.problem_type}")
