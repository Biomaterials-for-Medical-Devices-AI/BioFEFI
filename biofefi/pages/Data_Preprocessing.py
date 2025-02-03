from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from biofefi.components.experiments import experiment_selector
from biofefi.components.images.logos import sidebar_logo
from biofefi.options.choices import NORMALISATIONS, TRANSFORMATIONSY
from biofefi.options.enums import ConfigStateKeys, Normalisations, TransformationsY
from biofefi.options.file_paths import (
    biofefi_experiments_base_dir,
    data_preprocessing_options_path,
    execution_options_path,
    plot_options_path,
    raw_data_path,
)
from biofefi.options.preprocessing import PreprocessingOptions
from biofefi.services.configuration import (
    load_execution_options,
    load_plot_options,
    save_options,
)
from biofefi.services.experiments import get_experiments


def build_config() -> PreprocessingOptions:
    """
    Build the configuration object for preprocessing.
    """

    preprocessing_options = PreprocessingOptions(
        feature_selection_methods={
            ConfigStateKeys.VarianceThreshold: st.session_state[
                ConfigStateKeys.VarianceThreshold
            ],
            ConfigStateKeys.CorrelationThreshold: st.session_state[
                ConfigStateKeys.CorrelationThreshold
            ],
            ConfigStateKeys.LassoFeatureSelection: st.session_state[
                ConfigStateKeys.LassoFeatureSelection
            ],
        },
        dependent_variable_normalisation=st.session_state[
            ConfigStateKeys.DependentNormalisation
        ].lower(),
        independent_variable_transformation=st.session_state[
            ConfigStateKeys.IndependentNormalisation
        ].lower(),
    )
    return preprocessing_options


def run_feature_selection(feature_selection_methods: dict, data: pd.DataFrame) -> None:

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if feature_selection_methods[ConfigStateKeys.VarianceThreshold]:
        varianceselector = VarianceThreshold(
            threshold=st.session_state[ConfigStateKeys.ThresholdVariance]
        )
        X = varianceselector.fit_transform(X)
        variance_columns = varianceselector.get_feature_names_out()
        X = pd.DataFrame(X, columns=variance_columns)

    if feature_selection_methods[ConfigStateKeys.CorrelationThreshold]:
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column
            for column in upper_triangle.columns
            if any(
                upper_triangle[column]
                > st.session_state[ConfigStateKeys.ThresholdCorrelation]
            )
        ]
        X = X.drop(columns=to_drop)

    if feature_selection_methods[ConfigStateKeys.LassoFeatureSelection]:
        lasso = Lasso(alpha=st.session_state[ConfigStateKeys.RegularisationTerm])
        lasso.fit(X, y)
        selected_features = X.columns[lasso.coef_ != 0]
        X = X[selected_features]

    data = pd.concat([X, y], axis=1)


def normalise_independent_variables(normalisation_method: str, X):

    if normalisation_method == Normalisations.NoNormalisation:
        return X

    elif normalisation_method == Normalisations.Standardization:
        scaler = StandardScaler()

    elif normalisation_method == Normalisations.MinMax:
        scaler = MinMaxScaler()

    column_names = X.columns
    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=column_names)

    return X


def transform_dependent_variable(transformation_y_method: str, y):

    if transformation_y_method == TransformationsY.NoTransformation:
        return y

    column_name = y.name
    y = y.to_numpy().reshape(-1, 1)

    if transformation_y_method == TransformationsY.Log:
        y = np.log(y)

    elif transformation_y_method == TransformationsY.Sqrt:
        y = np.sqrt(y)

    elif transformation_y_method == TransformationsY.MinMaxNormalisation:
        scaler = MinMaxScaler()
        y = scaler.fit_transform(y)

    elif transformation_y_method == TransformationsY.StandardisationNormalisation:
        scaler = StandardScaler()
        y = scaler.fit_transform(y)

    y = pd.DataFrame(y, columns=[column_name])

    return y


def run_preprocessing(data: pd.DataFrame, experiment_path: Path) -> None:

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    config = build_config()
    save_options(data_preprocessing_options_path(experiment_path), config)
    X = normalise_independent_variables(config.dependent_variable_normalisation, X)
    y = transform_dependent_variable(config.independent_variable_transformation, y)

    data = pd.concat([X, y], axis=1)

    run_feature_selection(config.feature_selection_methods, data)

    return data


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

    exec_opt = load_execution_options(path_to_exec_opts)

    path_to_plot_opts = plot_options_path(
        biofefi_base_dir / st.session_state[ConfigStateKeys.ExperimentName]
    )

    path_to_raw_data = raw_data_path(
        exec_opt.data_path.split("/")[-1],
        biofefi_base_dir / st.session_state[ConfigStateKeys.ExperimentName],
    )

    if path_to_raw_data.exists():
        data = pd.read_csv(path_to_raw_data)
    else:
        data = pd.read_csv(exec_opt.data_path)

    plot_opt = load_plot_options(path_to_plot_opts)

    st.write("### Original Data")

    st.write(data)

    st.write("### Data Description")

    st.write(data.describe())

    st.write("## Data Preprocessing Options")

    st.write("#### Data Normalisation")

    st.write(
        """
        If you select **"Standardization"**, your data will be normalised by subtracting the
        mean and dividing by the standard deviation for each feature. The resulting transformation has a
        mean of 0 and values are between -1 and 1.

        If you select **"Minmax"**, your data will be scaled based on the minimum and maximum
        value of each feature. The resulting transformation will have values between 0 and 1.

        If you select **"None"**, the data will not be normalised.
        """
    )

    st.write("#### Normalisation Method for Independent Variables")

    st.selectbox(
        "Normalisation",
        NORMALISATIONS,
        key=ConfigStateKeys.DependentNormalisation,
        index=len(NORMALISATIONS) - 1,  # default to no normalisation
    )

    st.write("#### Transformation Method for Dependent Variable")

    transformationy = st.selectbox(
        "Transformations",
        TRANSFORMATIONSY,
        key=ConfigStateKeys.IndependentNormalisation,
        index=len(TRANSFORMATIONSY) - 1,  # default to no transformation
    )

    if (
        transformationy.lower() == TransformationsY.Log
        or transformationy.lower() == TransformationsY.Sqrt
    ):
        if (
            data.iloc[:, -1].min() <= 0
        ):  # deal with user attempting this transformations on negative values
            st.warning(
                "The dependent variable contains negative values. Log and square root transformations require positive values."
            )
            st.stop()

    st.write("#### Feature Selection")

    st.write("#### Check the Feature Selection Algorithms to Use")

    if st.checkbox(
        "Variance threshold",
        key=ConfigStateKeys.VarianceThreshold,
        help="Delete features with variance below a certain threshold",
    ):
        threshold = st.number_input(
            "Set threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            key=ConfigStateKeys.ThresholdVariance,
        )

    if st.checkbox(
        "Correlation threshold",
        key=ConfigStateKeys.CorrelationThreshold,
        help="Delete features with correlation above a certain threshold",
    ):
        threshold = st.number_input(
            "Set threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            key=ConfigStateKeys.ThresholdCorrelation,
        )

    if st.checkbox(
        "Lasso Feature Selection",
        key=ConfigStateKeys.LassoFeatureSelection,
        help="Select features using Lasso regression",
    ):
        regularisation_term = st.number_input(
            "Set regularisation term",
            min_value=0.0,
            value=0.05,
            key=ConfigStateKeys.RegularisationTerm,
        )

    if st.button("Run Data Preprocessing", type="primary"):

        data.to_csv(path_to_raw_data, index=False)

        data = run_preprocessing(
            data, biofefi_base_dir / st.session_state[ConfigStateKeys.ExperimentName]
        )

        data.to_csv(exec_opt.data_path, index=False)

        st.success("Data Preprocessing Complete")

        st.write("### Processed Data")

        st.write(data)

        st.write("### Processed Data Description")

        st.write(data.describe())
