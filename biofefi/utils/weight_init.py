import torch
import torch.nn.init as init


def normal_init(m: torch.nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    """
    Initializes the weights of Linear layers using a normal distribution.

    Parameters
    ----------
    m: torch.nn.Module
        The module to initialize
    mean: float
        The mean of the normal distribution
    std: float
        The standard deviation of the normal distribution
    """
    if isinstance(m, torch.nn.Linear):
        init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def xavier_init(m: torch.nn.Module) -> None:
    """
    Initializes the weights of Linear layers using Xavier initialization.

    Parameters
    ----------
    m: torch.nn.Module
        The module to initialize
    """
    if isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def kaiming_init(m: torch.nn.Module, nonlinearity: str = "relu") -> None:
    """
    Initializes the weights of Linear layers using Kaiming initialization.

    Parameters
    ----------
    m: torch.nn.Module
        The module to initialize
    nonlinearity: str
        The nonlinearity used in the network (e.g., 'relu', 'leaky_relu')
    """
    if isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None:
            init.constant_(m.bias, 0)