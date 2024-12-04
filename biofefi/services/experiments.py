import os
from pathlib import Path

from biofefi.options.execution import ExecutionOptions
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    execution_options_path,
    fi_options_dir,
    fi_options_path,
    fi_plot_dir,
    fi_result_dir,
    fuzzy_options_path,
    fuzzy_plot_dir,
    fuzzy_result_dir,
    log_dir,
    plot_options_path,
)
from biofefi.options.plotting import PlottingOptions
from biofefi.services.configuration import save_options
from biofefi.utils.utils import create_directory


def get_experiments() -> list[str]:
    """Get the list of experiments in the BioFEFI experiment directory.

    Returns:
        list[str]: The list of experiments.
    """
    # Get the base directory of all experiments
    base_dir = biofefi_experiments_base_dir()
    experiments = os.listdir(base_dir)
    # Filter out hidden files and directories
    experiments = filter(lambda x: not x.startswith("."), experiments)
    # Filter out files
    experiments = filter(
        lambda x: os.path.isdir(os.path.join(base_dir, x)), experiments
    )
    return experiments


def create_experiment(
    save_dir: Path,
    plotting_options: PlottingOptions,
    execution_options: ExecutionOptions,
):
    """Create an experiment on disk with it's global plotting options
    saved as a `json` file.

    Args:
        save_dir (Path): The path to where the experiment will be created.
        plotting_options (PlottingOptions): The plotting options to save.
    """
    create_directory(save_dir)
    plot_file_path = plot_options_path(save_dir)
    save_options(plot_file_path, plotting_options)
    execution_file_path = execution_options_path(save_dir)
    save_options(execution_file_path, execution_options)


def find_previous_fi_results(experiment_path: Path) -> bool:
    """Find previous feature importance results.

    Args:
        experiment_path (Path): The path to the experiment.

    Returns:
        bool: whether previous experiments exist or not.
    """

    previous_results = False

    directories = [
        fi_plot_dir(experiment_path),
        fi_result_dir(experiment_path),
        fi_options_dir(experiment_path),
        fuzzy_plot_dir(experiment_path),
        fuzzy_result_dir(experiment_path),
        fuzzy_options_path(experiment_path),
        fi_options_path(experiment_path),
        log_dir(experiment_path) / "fi",
        log_dir(experiment_path) / "fuzzy",
    ]

    for directory in directories:
        if directory.exists():
            previous_results = True
            break

    return previous_results
