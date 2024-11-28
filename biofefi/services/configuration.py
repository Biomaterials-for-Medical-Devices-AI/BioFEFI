import json, dataclasses
from pathlib import Path

from biofefi.options.execution import ExecutionOptions
from biofefi.options.fi import FeatureImportanceOptions
from biofefi.options.fuzzy import FuzzyOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions


def load_execution_options(path: Path) -> ExecutionOptions:
    """Load experiment execution options from the given path.
    The path will be to a `json` file containing the options.

    Args:
        path (Path): The path the `json` file containing the options.

    Returns:
        ExecutionOptions: The plotting options.
    """
    with open(path, "r") as json_file:
        options_json = json.load(json_file)
    options = ExecutionOptions(**options_json)
    return options


def save_options(
    path: Path,
    options: (
        ExecutionOptions
        | MachineLearningOptions
        | PlottingOptions
        | FeatureImportanceOptions
        | FuzzyOptions
    ),
):
    """Save options to a `json` file at the specified path.

    Args:
        path (Path): The path to the `json` file.
        options (ExecutionOptions  |  MachineLearningOptions  |  PlottingOptions  |  FeatureImportanceOptions  |  FuzzyOptions): The options to save.
    """
    options_json = dataclasses.asdict(options)
    with open(path, "w") as json_file:
        json.dump(options_json, json_file, indent=4)
