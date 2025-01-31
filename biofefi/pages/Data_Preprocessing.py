import pandas as pd
import streamlit as st

from biofefi.components.experiments import experiment_selector
from biofefi.components.images.logos import sidebar_logo
from biofefi.options.choices import NORMALISATIONS, TRANSFORMATIONSY
from biofefi.options.enums import ConfigStateKeys
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    execution_options_path,
    plot_options_path,
)
from biofefi.services.configuration import load_execution_options, load_plot_options
from biofefi.services.experiments import get_experiments


def run_preprocessing():
    pass


st.set_page_config(
    page_title="Data Preprocessing",
    page_icon=sidebar_logo(),
)

sidebar_logo()

st.header("Data Preprocessing")
st.write(
    """
    Here you can make changes to your data before running machine learning models. This includes feature selection and scalling of variables.
    """
)

choices = get_experiments()
experiment_name = experiment_selector(choices)
biofefi_base_dir = biofefi_experiments_base_dir()

if experiment_name:
    st.session_state[ConfigStateKeys.ExperimentName] = experiment_name

    path_to_exec_opts = execution_options_path(
        biofefi_base_dir / st.session_state[ConfigStateKeys.ExperimentName]
    )

    path_to_plot_opts = plot_options_path(
        biofefi_base_dir / st.session_state[ConfigStateKeys.ExperimentName]
    )

    exec_opt = load_execution_options(path_to_exec_opts)
    plot_opt = load_plot_options(path_to_plot_opts)

    data = pd.read_csv(exec_opt.data_path)

    st.write("### Original Data")

    st.write(data)

    st.write("### Data Description")

    st.write(data.describe())

    st.write("### Data Preprocessing Options")

    st.write("#### Feature Selection")

    st.toggle("Perform Feature Selection", key=ConfigStateKeys.FeatureSelection)

    st.write("### Data Normalisation")

    st.write("#### Normalisation Method for Independent Variables")

    st.selectbox(
        "Normalisation",
        NORMALISATIONS,
        key=ConfigStateKeys.DependentNormalisation,
    )

    st.write("#### Transformation Method for Dependent Variable")

    st.selectbox(
        "Normalisation",
        TRANSFORMATIONSY,
        key=ConfigStateKeys.IndependentNormalisation,
    )

    if st.button("Run Feature Importance", type="primary"):
        run_preprocessing()
