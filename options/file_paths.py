from pathlib import Path


def uploaded_file_path(file_name: str) -> str:
    """Create the full upload path for data file uploads.

    Args:
        file_name (str): The name of the file.

    Returns:
        str: The full upload path for the file.
    """
    return Path.home() / ".BioFEFIUploads" / file_name
