from pathlib import Path
import pytest

from biofefi.services.experiments import get_experiments
from biofefi.utils.utils import delete_directory


@pytest.fixture
def experiment_dir():
    # Arrange
    base_dir = Path.cwd() / "BioFEFIExperiments"
    base_dir.mkdir()
    experiment_dirs = ["experiment1", "experiment2"]
    for exp in experiment_dirs:
        directory = base_dir / exp
        directory.mkdir()
    yield base_dir, experiment_dirs

    # Cleanup
    if base_dir.exists():
        delete_directory(base_dir)


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
