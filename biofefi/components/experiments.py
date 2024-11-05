import streamlit as st
from pathlib import Path
from biofefi.options.enums import ViewExperimentKeys, ExplainModels
from biofefi.options.file_paths import biofefi_experiments_base_dir


def experiment_selector(options: list) -> Path:
    """Select

    Args:
        options (list): The list of experiment names to choose from.

    Returns:
        Path: The path to the experiment on disk.
    """

    return st.selectbox(
        "Select an experiment",
        options=options,
        index=None,
        placeholder="Experiment name",
        key=ViewExperimentKeys.ExperimentName,
    )


def model_selector(options: list) -> Path:
    """Select

    Args:
        options (list): The list of model names to choose from.

    Returns:
        Path: The path to the experiment on disk.
    """

    return st.multiselect(
        "Select an experiment",
        options=options,
        default=None,
        placeholder="Models to explain",
        key=ExplainModels.ExplainModels,
    )
