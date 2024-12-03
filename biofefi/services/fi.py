from typing import Any
import os
import pandas as pd
import shap
from pathlib import Path
import json
from biofefi.utils.logging_utils import Logger
from biofefi.utils.utils import delete_directory
from biofefi.options.file_paths import (
    log_dir,
    fi_plot_dir,
    fi_result_dir,
    fi_options_dir,
    fuzzy_plot_dir,
    fuzzy_result_dir,
    fi_options_path,
    fuzzy_options_path,
)


def calculate_global_shap_values(
    model,
    X: pd.DataFrame,
    shap_reduce_data: int,
    logger: Logger,
) -> tuple[pd.DataFrame, Any]:
    """Calculate SHAP values for a given model and dataset.

    Args:
        model: Model object.
        X (pd.DataFrame): The dataset.
        shap_reduce_data (int): The percentage of data to use for SHAP calculation.
        logger (Logger): The logger.

    Returns:
        tuple[pd.DataFrame, Any]: SHAP dataframe and SHAP values.
    """
    logger.info(f"Calculating SHAP Importance for {model.__class__.__name__} model..")

    if shap_reduce_data == 100:
        explainer = shap.Explainer(model.predict, X)
    else:
        X_reduced = shap.utils.sample(X, int(X.shape[0] * shap_reduce_data / 100))
        explainer = shap.Explainer(model.predict, X_reduced)

    shap_values = explainer(X)

    # Calculate Average Importance + set column names as index
    shap_df = (
        pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().to_frame()
    )

    logger.info("SHAP Importance Analysis Completed..")

    # Return the DataFrame
    return shap_df, shap_values


def calculate_local_shap_values(
    model,
    X: pd.DataFrame,
    shap_reduce_data: int,
    logger: Logger,
) -> tuple[pd.DataFrame, Any]:
    """Calculate local SHAP values for a given model and dataset.

    Args:
        model: Model object.
        X (pd.DataFrame): The dataset.
        shap_reduce_data (int): The percentage of data to use for SHAP calculation.
        logger (Logger): The logger.

    Returns:
        tuple[pd.DataFrame, Any]: SHAP dataframe and SHAP values.
    """
    logger.info(f"Calculating SHAP Importance for {model.__class__.__name__} model..")

    if shap_reduce_data == 100:
        explainer = shap.Explainer(model.predict, X)
    else:
        X_reduced = shap.utils.sample(X, int(X.shape[0] * shap_reduce_data / 100))
        explainer = shap.Explainer(model.predict, X_reduced)

    shap_values = explainer(X)

    shap_df = pd.DataFrame(shap_values.values, columns=X.columns, index=X.index)
    # TODO: scale coefficients between 0 and +1 (low to high impact)

    logger.info("SHAP Importance Analysis Completed..")

    # Return the DataFrame
    return shap_df, shap_values


def load_fi_options(path: Path) -> dict:
    """Load feature importance options.

    Args:
        path (Path): The path to the feature importance options file.

    Returns:
        dict: The feature importance options.
    """

    try:
        with open(path, "r") as file:
            fi_options = json.load(file)
    except FileNotFoundError:
        fi_options = None

    return fi_options


def delete_previous_FI_results(experiment_path: Path):
    """Delete previous feature importance results.

    Args:
        experiment_path (Path): The path to the experiment.
    """
    if os.path.exists(fi_plot_dir(experiment_path)):
        delete_directory(fi_plot_dir(experiment_path))
    if os.path.exists(fi_result_dir(experiment_path)):
        delete_directory(fi_result_dir(experiment_path))
    if os.path.exists(fi_options_dir(experiment_path)):
        delete_directory(fi_options_dir(experiment_path))
    if os.path.exists(fuzzy_plot_dir(experiment_path)):
        delete_directory(fuzzy_plot_dir(experiment_path))
    if os.path.exists(fuzzy_result_dir(experiment_path)):
        delete_directory(fuzzy_result_dir(experiment_path))
    if os.path.exists(fuzzy_options_path(experiment_path)):
        delete_directory(fuzzy_options_path(experiment_path))
    if os.path.exists(fi_options_path(experiment_path)):
        delete_directory(fi_options_path(experiment_path))
    if os.path.exists(log_dir(experiment_path) / "fi"):
        delete_directory(log_dir(experiment_path) / "fi")
    if os.path.exists(log_dir(experiment_path) / "fuzzy"):
        delete_directory(log_dir(experiment_path) / "fuzzy")
