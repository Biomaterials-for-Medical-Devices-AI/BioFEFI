import streamlit as st

from options.enums import ConfigStateKeys


def data_upload_form():
    st.header("Data Upload")
    st.text_input("Name of the experiment", key=ConfigStateKeys.ExperimentName)
    st.text_input(
        "Name of the dependent variable", key=ConfigStateKeys.DependentVariableName
    )
    st.file_uploader(
        "Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName
    )
    st.button("Run", key=ConfigStateKeys.RunPipeline)
