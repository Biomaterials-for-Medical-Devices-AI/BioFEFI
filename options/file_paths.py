from pathlib import Path
import os

BASE_DIR = os.getenv("BASE_DIR", Path.home() / ".BioFEFI")
if not isinstance(BASE_DIR, Path):
    BASE_DIR = Path(BASE_DIR)


def uploaded_file_path(file_name: str, experiment_path: Path) -> Path:
    """Create the full upload path for data file uploads.

    Args:
        file_name (str): The name of the file.
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full upload path for the file.
    """
    return experiment_path / file_name


def log_dir(experiment_path: Path) -> Path:
    """Create the full upload path for experiment log files.

    Args:
        experiment_path (str): The path of the experiment.

    Returns:
        Path: The full path for the log directory.
    """
    return experiment_path / "logs"


def ml_plot_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Machine Learning plots.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Machine Learning plot directory.
    """
    return experiment_path / "plots" / "ml"


def ml_model_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Machine Learning models.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Machine Learning model directory.
    """
    return experiment_path / "models"


def fi_plot_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Feature Importance plots.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Feature Importance plot directory.
    """
    return experiment_path / "plots" / "fi"


def fi_result_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Feature Importance results.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Feature Importance result directory.
    """
    return experiment_path / "results" / "fi"


def fuzzy_plot_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Fuzzy plots.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Fuzzy plot directory.
    """
    return experiment_path / "plots" / "fuzzy"


def fuzzy_result_dir(experiment_path: str) -> Path:
    """Create the full path to the directory to save Fuzzy results.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Fuzzy result directory.
    """
    return experiment_path / "results" / "fuzzy"
