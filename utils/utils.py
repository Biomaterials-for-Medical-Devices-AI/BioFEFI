import inspect
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def set_seed(seed: int) -> None:
    """
    Sets the seed for the experiment

    Parameters
    ----------
    seed: int
        The seed to use for the experiment
    """
    # torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.cuda.manual_seed(seed)


def assert_model_param(model, model_params, logger: object = None) -> None:
    """
    Asserts that the model parameters are valid

    Parameters
    ----------
    model: object
        The model to check the parameters for
    model_params: dict
        The model parameters to check
    logger: object
        The logger to use for logging
    """
    if hasattr(model, "args"):
        original_args = model.args
    else:
        original_args = list(inspect.signature(model).parameters)

    args_to_remove = []
    for arg in model_params:
        if arg not in original_args:
            logger.warning(
                f"Model {model.__name__} does not have parameter {arg}, removing"
            )
            args_to_remove.append(arg)
    for arg in args_to_remove:
        model_params.pop(arg)
    logger.info(f"Using model {model.__name__} with parameters {model_params}")
    return model_params
