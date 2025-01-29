from pathlib import Path

from biofefi.options.execution import ExecutionOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.services.experiments import create_experiment, get_experiments

# import all the fixtures for services
from .fixtures import *  # noqa: F403, F401


def test_get_experiments_with_base_dir(experiment_dir: tuple[Path, list[str]]):
    # Arrange
    base_dir, expected_experiments = experiment_dir

    # Act
    actual_experiments = get_experiments(base_dir)

    # Assert
    assert isinstance(actual_experiments, list)
    assert actual_experiments == expected_experiments


def test_get_experiments_without_base_dir():
    # Act
    actual_experiments = get_experiments()

    # Assert
    assert isinstance(actual_experiments, list)


def test_create_experiment(
    experiment_dir, execution_opts: ExecutionOptions, plotting_opts: PlottingOptions
):
    # Arrange
    base_dir, experiments = experiment_dir
    save_dir = base_dir / experiments[0]  # use the first experiment directory

    # don't use the functions `execution_options_path` and `plot_options_path`
    # this would create coupling with those tests
    execution_options_file = save_dir / "execution_options.json"
    plotting_options_file = save_dir / "plot_options.json"

    # Act
    create_experiment(save_dir, plotting_opts, execution_opts)

    # Assert
    assert execution_options_file.exists()
    assert plotting_options_file.exists()
