from pathlib import Path


def uploaded_file_path(file_name: str, exeperiment_name: str) -> Path:
    """Create the full upload path for data file uploads.

    Args:
        file_name (str): The name of the file.
        exeperiment_name (str): The name of the experiment. This will be used
        to create a subdirectory with the this value.

    Returns:
        Path: The full upload path for the file.
    """
    return Path.home() / ".BioFEFIUploads" / exeperiment_name / file_name
