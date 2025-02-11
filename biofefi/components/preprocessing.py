import pandas as pd
import streamlit as st


@st.experimental_fragment
def preprocessed_view(data: pd.DataFrame):
    """Display the preprocessed data to the user.

    Args:
        data (pd.DataFrame): The preprocessed data to show.
    """
    st.write("### Processed Data")
    st.write(data)
    st.write("### Processed Data Description")
    st.write(data.describe())
