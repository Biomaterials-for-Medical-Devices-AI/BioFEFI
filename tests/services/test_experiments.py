from pathlib import Path

from biofefi.services.experiments import get_experiments

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
