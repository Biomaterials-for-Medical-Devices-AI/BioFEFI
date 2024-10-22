from pathlib import Path
import streamlit as st

from options.enums import ConfigStateKeys, ExecutionStateKeys


def data_upload_form():
    st.header("Data Upload")
    save_dir = _save_directory_selector()
    if not _directory_is_valid(save_dir):
        st.markdown(f":red[Cannot use {save_dir}; it already exists.]")
    else:
        st.session_state[ConfigStateKeys.SaveDir] = save_dir
    st.text_input("Name of the experiment", key=ConfigStateKeys.ExperimentName)
    st.text_input(
        "Name of the dependent variable", key=ConfigStateKeys.DependentVariableName
    )
    st.file_uploader(
        "Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName
    )
    if not st.session_state.get(ConfigStateKeys.IsMachineLearning, False):
        st.file_uploader(
            "Upload machine leaerning models",
            type="pkl",
            accept_multiple_files=True,
            key=ConfigStateKeys.UploadedModels,
        )
    st.button("Run", key=ExecutionStateKeys.RunPipeline)


def _save_directory_selector() -> Path:
    """Create a selector for the directory to save experiments."""
    root = Path.home()

    col1, col2 = st.columns([0.3, 0.7], vertical_alignment="bottom")

    col1.text(f"{root}/")
    sub_dir = col2.text_input(label="", placeholder="Directory name")

    return root / sub_dir


def _directory_is_valid(directory: Path) -> bool:
    """Determine if the directory supplied by the user is valid. If it already exists,
    it is invalid.

    Args:
        directory (Path): The path to check.

    Returns:
        bool: `True` if the directory doesn't already exist, else `False`
    """
    return not directory.exists()
