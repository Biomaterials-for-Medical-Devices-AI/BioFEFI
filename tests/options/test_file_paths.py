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
