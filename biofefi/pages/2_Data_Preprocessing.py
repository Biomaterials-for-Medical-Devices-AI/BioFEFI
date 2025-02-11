from pathlib import Path
import pandas as pd
import streamlit as st

from biofefi.components.experiments import experiment_selector
from biofefi.components.forms import preprocessing_opts_form
from biofefi.components.images.logos import sidebar_logo
from biofefi.components.preprocessing import original_view, preprocessed_view
from biofefi.options.enums import (
    DataPreprocessingStateKeys,
    ExecutionStateKeys,
)
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    execution_options_path,
    plot_options_path,
    preprocessed_data_path,
)
from biofefi.options.preprocessing import PreprocessingOptions
from biofefi.services.configuration import (
    load_execution_options,
    load_plot_options,
    save_options,
)
from biofefi.services.experiments import get_experiments
from biofefi.services.preprocessing import run_preprocessing


def build_config() -> PreprocessingOptions:
    """
    Build the configuration object for preprocessing.
    """

    preprocessing_options = PreprocessingOptions(
        feature_selection_methods={
            DataPreprocessingStateKeys.VarianceThreshold: st.session_state[
                DataPreprocessingStateKeys.VarianceThreshold
            ],
            DataPreprocessingStateKeys.CorrelationThreshold: st.session_state[
                DataPreprocessingStateKeys.CorrelationThreshold
            ],
            DataPreprocessingStateKeys.LassoFeatureSelection: st.session_state[
                DataPreprocessingStateKeys.LassoFeatureSelection
            ],
        },
        variance_threshold=st.session_state[
            DataPreprocessingStateKeys.ThresholdVariance
        ],
        correlation_threshold=st.session_state[
            DataPreprocessingStateKeys.ThresholdCorrelation
        ],
        lasso_regularisation_term=st.session_state[
            DataPreprocessingStateKeys.RegularisationTerm
        ],
        independent_variable_normalisation=st.session_state[
            DataPreprocessingStateKeys.IndependentNormalisation
        ].lower(),
        dependent_variable_transformation=st.session_state[
            DataPreprocessingStateKeys.DependentNormalisation
        ].lower(),
    )
    return preprocessing_options


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
    st.session_state[ExecutionStateKeys.ExperimentName] = experiment_name

    path_to_exec_opts = execution_options_path(biofefi_base_dir / experiment_name)

    exec_opt = load_execution_options(path_to_exec_opts)

    path_to_plot_opts = plot_options_path(biofefi_base_dir / experiment_name)

    # Check if the user has already preprocessed their data
    if exec_opt.data_is_preprocessed:
        st.warning("Your data are already preprocessed. Would you like to start again?")
        preproc_again = st.checkbox("Redo preprocessing", value=False)
    else:
        preproc_again = False

    if preproc_again:
        # remove preprocessed suffix
        exec_opt.data_path = exec_opt.data_path.replace("_preprocessed", "")
        # set data_is_preprocessed to False
        exec_opt.data_is_preprocessed = False

    data = pd.read_csv(exec_opt.data_path)

    path_to_preprocessed_data = preprocessed_data_path(
        Path(exec_opt.data_path).name,
        biofefi_base_dir / experiment_name,
    )

    plot_opt = load_plot_options(path_to_plot_opts)

    original_view(data)

    preprocessing_opts_form(data)

    if st.button("Run Data Preprocessing", type="primary"):

        config = build_config()

        processed_data = run_preprocessing(
            data,
            biofefi_base_dir / experiment_name,
            config,
        )

        processed_data.to_csv(path_to_preprocessed_data, index=False)

        # Update exec opts to point to the pre-processed data
        exec_opt.data_path = str(path_to_preprocessed_data)
        # Set data_is_preprocessed to True
        exec_opt.data_is_preprocessed = True
        save_options(path_to_exec_opts, exec_opt)

        st.success("Data Preprocessing Complete")
        preprocessed_view(processed_data)
