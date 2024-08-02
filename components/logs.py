import streamlit as st
from ..options.enums import ConfigStateKeys


def log_box(value: str) -> str:
    """Display a text area which shows that logs of the current pipeline run.

    Args:
        value (str): The text to show in the box.

    Returns:
        str: The text in the box.
    """
    return st.text_area("Logs", value=value, disabled=True, key=ConfigStateKeys.LogBox)
