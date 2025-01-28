from pathlib import Path
import pytest

from biofefi.options.execution import ExecutionOptions
from biofefi.options.choices import DATA_SPLITS
from biofefi.options.fi import FeatureImportanceOptions
from biofefi.options.file_paths import (
    execution_options_path,
    fi_options_path,
    fuzzy_options_path,
    ml_options_path,
    plot_options_path,
)
from biofefi.options.fuzzy import FuzzyOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.services.configuration import save_options


@pytest.fixture
def execution_opts():
    # Arrange
    return ExecutionOptions(data_path="test_data.csv", data_split=DATA_SPLITS[0])


@pytest.fixture
def execution_opts_file():
    # Arrange
    experiment_path = Path.cwd()
    options_file = execution_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


@pytest.fixture
def plotting_opts():
    # Arrange
    return PlottingOptions(
        plot_axis_font_size=8,
        plot_axis_tick_size=5,
        plot_colour_scheme="fancy_colours",  # not real but doesn't matter for testing
        angle_rotate_xaxis_labels=0,
        angle_rotate_yaxis_labels=0,
        save_plots=True,
        plot_font_family="sans-serif",
        plot_title_font_size=14,
    )


@pytest.fixture
def plotting_opts_file():
    # Arrange
    experiment_path = Path.cwd()
    options_file = plot_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


@pytest.fixture
def ml_opts():
    # Arrange
    return MachineLearningOptions(
        model_types={}
    )  # no need to specify any model types, use empty dict


@pytest.fixture
def ml_opts_file():
    # Arrange
    experiment_path = Path.cwd()
    options_file = ml_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


@pytest.fixture
def fi_opts():
    # Arrange
    return FeatureImportanceOptions(
        global_importance_methods={},
        feature_importance_ensemble={},
        local_importance_methods={},
    )  # no need to specify any of these, use empty dict


@pytest.fixture
def fi_opts_file():
    # Arrange
    experiment_path = Path.cwd()
    options_file = fi_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


@pytest.fixture
def fuzzy_opts():
    # Arrange
    return FuzzyOptions(cluster_names=[])  # no need to specify this, use empty list


@pytest.fixture
def fuzzy_opts_file():
    # Arrange
    experiment_path = Path.cwd()
    options_file = fuzzy_options_path(experiment_path)
    yield options_file

    # Cleanup
    if options_file.exists():
        options_file.unlink()


def test_save_execution_opts(execution_opts, execution_opts_file):
    # Act
    save_options(execution_opts_file, execution_opts)

    # Assert
    assert execution_opts_file.exists()


def test_save_plotting_opts(plotting_opts, plotting_opts_file):
    # Act
    save_options(plotting_opts_file, plotting_opts)

    # Assert
    assert plotting_opts_file.exists()


def test_save_ml_opts(ml_opts, ml_opts_file):
    # Act
    save_options(ml_opts_file, ml_opts)

    # Assert
    assert ml_opts_file.exists()


def test_save_fi_opts(fi_opts, fi_opts_file):
    # Act
    save_options(fi_opts_file, fi_opts)

    # Assert
    assert fi_opts_file.exists()


def test_save_fuzzy_opts(fuzzy_opts, fuzzy_opts_file):
    # Act
    save_options(fuzzy_opts_file, fuzzy_opts)

    # Assert
    assert fuzzy_opts_file.exists()
