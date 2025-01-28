from pathlib import Path
import biofefi.options.file_paths as fp


def test_biofefi_experiments_base_dir():
    # Arrange
    expected_output = Path.home() / "BioFEFIExperiments"

    # Act
    actual_output = fp.biofefi_experiments_base_dir()

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_uploaded_file_path():
    # Arrange
    experiment_path = fp.biofefi_experiments_base_dir() / "TestExperiment"
    file_name = "test_data.csv"
    expected_output = experiment_path / file_name

    # Act
    actual_output = fp.uploaded_file_path(file_name, experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_log_dir():
    # Arrange
    experiment_path = fp.biofefi_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "logs"

    # Act
    actual_output = fp.log_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output


def test_ml_plot_dir():
    # Arrange
    experiment_path = fp.biofefi_experiments_base_dir() / "TestExperiment"
    expected_output = experiment_path / "plots" / "ml"

    # Act
    actual_output = fp.ml_plot_dir(experiment_path)

    # Assert
    assert isinstance(actual_output, Path)
    assert actual_output == expected_output
