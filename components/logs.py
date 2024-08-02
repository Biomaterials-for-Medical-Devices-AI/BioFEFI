import streamlit as st
from options.enums import ConfigStateKeys


def log_box() -> str:
    """Display a text area which shows that logs of the current pipeline run.

    Returns:
        str: The text in the box.
    """
    return st.text_area(
        "Logs",
        # value=st.session_state.get(ConfigStateKeys.LogBox, ""),
        key=ConfigStateKeys.LogBox,
    )
