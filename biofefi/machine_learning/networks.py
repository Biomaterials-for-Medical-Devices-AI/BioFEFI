import os
import sys
import torch

import torch.nn as nn
from typing import Tuple
from biofefi.utils.weight_init import normal_init, xavier_init, kaiming_init

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseNetwork(nn.Module):
    """
    This class is an abstract class for networks
    """

    def __init__(self, opt) -> None:
        """
        Initializes the BaseNetwork class
        """
        super().__init__()
        self._name = "BaseNetwork"
        self._opt = opt
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def name(self) -> str:
        """
        Returns the name of the network
        """
        return self._name

    def _make_loss(self, problem_type):
        """
        Creates the loss function
        """
        if problem_type == "classification":
            self.loss = nn.CrossEntropyLoss()
        elif problem_type == "regression":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f"Problem type {problem_type} not implemented")

    def _initialise_weights(self, init_type: str = "normal") -> None:
        """
        Initializes the weights of the network

        Parameters
        ----------
        init_type: str
            The type of initialization

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        if init_type == "normal":
            self.apply(normal_init)
        elif init_type == "xavier_normal":
            self.apply(xavier_init)
        elif init_type == "kaiming_normal":
            self.apply(kaiming_init)
        else:
            raise NotImplementedError(f"Invalid init type: {init_type}")

    def _make_optimizer(self, optimizer_type, lr):
        """
        Creates the optimizer for the network

        Parameters
        ----------
        optimizer_type: str
            The type of optimizer to use

        lr: float
            The learning rate for the optimizer

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer_type == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError(
                f"Optimizer type {optimizer_type} not implemented"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the network

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Returns the string representation of the network
        """
        return self._name

    def get_num_params(self) -> Tuple[int, int]:
        """
        Returns the number of parameters in the network

        Returns
        -------
        all_params: int
            The total number of parameters in the network
        trainable_params: int
            The total number of trainable parameters in the network
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return all_params, trainable_params

    def save_model(self):
        """
        Saves the model

        Returns
        -------
        None
        """
        try:
            torch.save(
                self.state_dict(),
                os.path.join(self._opt.checkpoints_dir, f"{self._name}.pth"),
            )

        except Exception as e:
            raise NotImplementedError(f"Method not implemented: {e}")
