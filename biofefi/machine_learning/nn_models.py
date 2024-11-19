import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from biofefi.machine_learning.networks import BaseNetwork
from biofefi.options.enums import ProblemTypes, ModelNames, OptimiserTypes


class BayesianRegularisedNeuralNets(
    BaseNetwork, BaseEstimator, ClassifierMixin, RegressorMixin
):
    """
    This class implements Bayesian Regularised Neural Networks
    compatible with scikit-learn.
    """

    def __init__(self, opt):
        """
        Initializes the BayesianRegularisedNeuralNets class.

        Args:
            opt (argparse.Namespace): The options for the
            BayesianRegularisedNeuralNets class, typically
            passed from a command-line interface.
        """
        super().__init__(opt)
        self._name = ModelNames.BayesianRegularisedNeuralNets
        self._opt = opt
        self._hidden_dim = opt.hidden_dim

        self.problem_type = opt.problem_type

        # Layers will be initialized in `_initialize_network`
        self.layer1 = None
        self.layer2 = None
        self.output_layer = None

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
        self.layer1 = nn.Linear(input_dim, self._hidden_dim)
        self.layer2 = nn.Linear(self._hidden_dim, self._hidden_dim)
        self.output_layer = nn.Linear(self._hidden_dim, output_dim)

        # Initialize weights and create optimizer here after layers are defined
        self._initialise_weights()
        self._make_optimizer(OptimiserTypes.Adam, self._opt.lr)

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
        if self.problem_type.lower() == ProblemTypes.Classification:

            # Use softmax for multi-class classification, sigmoid for binary
            return torch.sigmoid(x) if x.size(1) == 1 else torch.softmax(x, dim=1)

        # No activation for regression
        elif self.problem_type.lower() == ProblemTypes.Regression:
            return x
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    def bayesian_regularization_loss(
        self, prior_mu: float = None, prior_sigma: float = None
    ) -> torch.Tensor:
        """
        Compute the Bayesian Regularization loss.

        The loss is computed as the sum of squared differences
        between model parameters and their prior mean,
        scaled by the prior standard deviation.

        Args:
            prior_mu (float, optional): The prior mean. Defaults
            to `self._opt.prior_mu` if not provided.

            prior_sigma (float, optional): The prior standard deviation.
            Defaults to `self._opt.prior_sigma` if not provided.

        Returns:
            torch.Tensor: The computed regularization loss.

        Raises:
            ValueError: If both `prior_mu` and `prior_sigma` are not provided.
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
        Compute the total loss based on the problem type
        and include regularization loss.

        Args:
            outputs (torch.Tensor): The predicted outputs from the model.
            targets (torch.Tensor): The true target values.

        Returns:
            torch.Tensor: The total computed loss, including both
            predictive and regularization loss.

        Raises:
            ValueError: If an unsupported problem type is specified.
        """

        # Determine the predictive loss based on problem type
        if self.problem_type.lower() == ProblemTypes.Classification:
            if outputs.size(1) == 1:
                predictive_loss = nn.BCELoss()(outputs, targets)

            else:
                predictive_loss = nn.CrossEntropyLoss()(outputs, targets)
                print("Using CrossEntropyLoss for multi-class classification.")
        elif self.problem_type.lower() == ProblemTypes.Regression:
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
        Train the Bayesian Regularized Neural Network.

        Args:
            X (np.ndarray): The input training data.
            y (np.ndarray): The target training data.

            validation_data (Optional[Tuple[np.ndarray, np.ndarray]]): Optional
            validation data (inputs, targets).

        Returns:
            self: The trained model.

        """
        # Convert input data to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Adjust target `y` for classification tasks
        if self.problem_type.lower() == ProblemTypes.Classification:

            # Ensure y is 1D and of type Long for CrossEntropyLoss
            y = y.squeeze().long()

        # Convert validation data if provided
        if validation_data is not None:
            val_X, val_y = validation_data
            val_X = torch.tensor(val_X, dtype=torch.float32)
            val_y = torch.tensor(val_y, dtype=torch.float32)

            # Ensure validation y is also 1D for classification
            if self.problem_type.lower() == ProblemTypes.Classification:
                val_y = val_y.squeeze().long()
            validation_data = (val_X, val_y)

        input_dim = X.shape[1]

        # Determine number of classes if classification
        if self.problem_type.lower() == ProblemTypes.Classification:

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

        Args:
            X (np.ndarray): The input data features for prediction.

        Returns:
            np.ndarray: The predicted outputs as a numpy array.

        Raises:
            ValueError: If an unsupported problem type is specified.
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            if self.problem_type.lower() == ProblemTypes.Classification:
                if outputs.size(1) == 1:
                    return (
                        (outputs > self._opt.classification_cutoff)
                        .float()
                        .cpu()
                        .numpy()
                    )
                else:
                    return torch.argmax(outputs, dim=1).cpu().numpy()
            elif self.problem_type.lower() == ProblemTypes.Regression:
                return outputs.cpu().numpy()
            else:
                raise ValueError(f"Unsupported problem type: {self.problem_type}")
