import json, dataclasses
from pathlib import Path

from biofefi.options.plotting import PlottingOptions


def save_plot_options(path: Path, options: PlottingOptions):
    """Save plot options to a `json` file at the specified path.

    Args:
        path (Path): The path to the `json` file.
        options (PlottingOptions): The options to save.
    """
    options_json = dataclasses.asdict(options)
    with open(path, "w") as json_file:
        json.dump(options_json, json_file)
