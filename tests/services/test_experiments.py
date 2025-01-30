from pathlib import Path

from biofefi.options.execution import ExecutionOptions
from biofefi.options.file_paths import (
    execution_options_path,
    fi_result_dir,
    plot_options_path,
)
from biofefi.options.plotting import PlottingOptions
from biofefi.services.experiments import (
    create_experiment,
    find_previous_fi_results,
    get_experiments,
)

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
    experiment_dir: tuple[Path, list[str]],
    execution_opts: ExecutionOptions,
    plotting_opts: PlottingOptions,
):
    # Arrange
    base_dir, experiments = experiment_dir
    save_dir = base_dir / experiments[0]  # use the first experiment directory

    execution_options_file = execution_options_path(save_dir)
    plotting_options_file = plot_options_path(save_dir)

    # Act
    create_experiment(save_dir, plotting_opts, execution_opts)

    # Assert
    assert execution_options_file.exists()
    assert execution_options_file.is_file()
    assert plotting_options_file.exists()
    assert plotting_options_file.is_file()


def test_find_previous_fi_results_when_empty(
    experiment_dir: tuple[Path, list[str]],
):
    # Arrange
    base_dir, experiments = experiment_dir
    exp_dir = base_dir / experiments[0]  # use the first experiment directory

    # Act
    results_found = find_previous_fi_results(exp_dir)

    # Assert
    assert not results_found


def test_find_previous_fi_results(
    previous_fi_results: Path,
):
    # Act
    results_found = find_previous_fi_results(previous_fi_results)

    # Assert
    assert results_found
