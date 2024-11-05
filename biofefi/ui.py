from argparse import Namespace
from multiprocessing import Process
from biofefi.components.images.logos import header_logo, sidebar_logo
from biofefi.components.logs import log_box
from biofefi.components.forms import data_upload_form
from biofefi.components.plots import plot_box
from biofefi.components.configuration import (
    ml_options_box,
    plot_options_box,
    fi_options_box,
)
from biofefi.options.execution import ExecutionOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.services.logs import get_logs
from biofefi.services.ml_models import save_model, load_models
from biofefi.feature_importance import feature_importance, fuzzy_interpretation
from biofefi.feature_importance.feature_importance_options import (
    FeatureImportanceOptions,
)
from biofefi.feature_importance.fuzzy_options import FuzzyOptions
from biofefi.machine_learning import train
from biofefi.machine_learning.data import DataBuilder
from biofefi.machine_learning.ml_options import MLOptions
from biofefi.options.enums import (
    ConfigStateKeys,
    ExecutionStateKeys,
    ProblemTypes,
    PlotOptionKeys,
)
from biofefi.options.file_paths import (
    fi_plot_dir,
    fuzzy_plot_dir,
    uploaded_file_path,
    log_dir,
    ml_plot_dir,
    ml_model_dir,
)
from biofefi.utils.logging_utils import Logger, close_logger
from biofefi.utils.utils import set_seed
import streamlit as st
import os


def build_configuration() -> (
    tuple[Namespace, Namespace, MachineLearningOptions, ExecutionOptions, str]
):
    """Build the configuration objects for the pipeline.

    Returns:
        tuple[Namespace, Namespace, MachineLearningOptions, ExecutionOptions, str]: The configuration for fuzzy, FI and ML pipelines, the general execution and the experiment name.
    """

    fuzzy_opt = FuzzyOptions()
    fuzzy_opt.initialize()
    if st.session_state.get(ConfigStateKeys.FuzzyFeatureSelection, False):
        fuzzy_opt.parser.set_defaults(
            fuzzy_feature_selection=st.session_state[
                ConfigStateKeys.FuzzyFeatureSelection
            ],
            num_fuzzy_features=st.session_state[ConfigStateKeys.NumberOfFuzzyFeatures],
            granular_features=st.session_state[ConfigStateKeys.GranularFeatures],
            num_clusters=st.session_state[ConfigStateKeys.NumberOfClusters],
            cluster_names=st.session_state[ConfigStateKeys.ClusterNames],
            num_rules=st.session_state[ConfigStateKeys.NumberOfTopRules],
            save_fuzzy_set_plots=st.session_state[PlotOptionKeys.SavePlots],
            # fuzzy_log_dir=
            dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
            experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
            problem_type=st.session_state.get(
                ConfigStateKeys.ProblemType, ProblemTypes.Auto
            ).lower(),
            is_granularity=st.session_state[ConfigStateKeys.GranularFeatures],
        )
    fuzzy_opt = fuzzy_opt.parse()

    fi_opt = FeatureImportanceOptions()
    fi_opt.initialize()
    if st.session_state.get(ConfigStateKeys.IsFeatureImportance, False):
        fi_opt.parser.set_defaults(
            num_features_to_plot=st.session_state[
                ConfigStateKeys.NumberOfImportantFeatures
            ],
            permutation_importance_scoring=st.session_state[
                ConfigStateKeys.ScoringFunction
            ],
            permutation_importance_repeat=st.session_state[
                ConfigStateKeys.NumberOfRepetitions
            ],
            shap_reduce_data=st.session_state[ConfigStateKeys.ShapDataPercentage],
            dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
            experiment_name=st.session_state[ConfigStateKeys.ExperimentName],
            problem_type=st.session_state.get(
                ConfigStateKeys.ProblemType, ProblemTypes.Auto
            ).lower(),
            is_feature_importance=st.session_state[ConfigStateKeys.IsFeatureImportance],
            # fi_log_dir=
            angle_rotate_xaxis_labels=st.session_state[
                PlotOptionKeys.RotateXAxisLabels
            ],
            angle_rotate_yaxis_labels=st.session_state[
                PlotOptionKeys.RotateYAxisLabels
            ],
            save_feature_importance_plots=st.session_state[PlotOptionKeys.SavePlots],
            save_feature_importance_options=st.session_state[
                ConfigStateKeys.SaveFeatureImportanceOptions
            ],
            save_feature_importance_results=st.session_state[
                ConfigStateKeys.SaveFeatureImportanceResults
            ],
            local_importance_methods=st.session_state[
                ConfigStateKeys.LocalImportanceFeatures
            ],
            feature_importance_ensemble=st.session_state[
                ConfigStateKeys.EnsembleMethods
            ],
            global_importance_methods=st.session_state[
                ConfigStateKeys.GlobalFeatureImportanceMethods
            ],
        )
    fi_opt = fi_opt.parse()

    path_to_data = uploaded_file_path(
        st.session_state[ConfigStateKeys.UploadedFileName].name,
        st.session_state[ConfigStateKeys.ExperimentName],
    )

    exec_opts = ExecutionOptions(
        data_path=path_to_data,
        experiment_name=ConfigStateKeys.ExperimentName,
        is_feature_engineering=False,  # not implemented so False for now
        is_machine_learning=st.session_state[ConfigStateKeys.IsMachineLearning],
        is_feature_importance=st.session_state[ConfigStateKeys.IsFeatureImportance],
        random_state=st.session_state[ConfigStateKeys.RandomSeed],
        problem_type=st.session_state.get(
            ConfigStateKeys.ProblemType, ProblemTypes.Auto
        ).lower(),
        dependent_variable=st.session_state[ConfigStateKeys.DependentVariableName],
        normalization=st.session_state[ConfigStateKeys.Normalization],
    )

    ml_opt = MachineLearningOptions(
        data_split=st.session_state[ConfigStateKeys.DataSplit],
        n_bootstraps=st.session_state[ConfigStateKeys.NumberOfBootstraps],
        save_actual_pred_plots=st.session_state[PlotOptionKeys.SavePlots],
        model_types=st.session_state[ConfigStateKeys.ModelTypes],
        ml_log_dir=ml_plot_dir(st.session_state[ConfigStateKeys.ExperimentName]),
        save_models=st.session_state[ConfigStateKeys.SaveModels],
    )

    return (
        fuzzy_opt,
        fi_opt,
        ml_opt,
        exec_opts,
        st.session_state[ConfigStateKeys.ExperimentName],
    )


def save_upload(file_to_upload: str, content: str, mode: str = "w"):
    """Save a file given to the UI to disk.

    Args:
        file_to_upload (str): The name of the file to save.
        content (str): The contents to save to the file.
        mode (str): The mode to write the file. e.g. "w", "w+", "wb", etc.
    """
    base_dir = os.path.dirname(file_to_upload)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with open(file_to_upload, mode) as f:
        f.write(content)


def pipeline(
    fuzzy_opts: Namespace,
    fi_opts: Namespace,
    ml_opts: MachineLearningOptions,
    exec_opts: ExecutionOptions,
    experiment_name: str,
):
    """This function actually performs the steps of the pipeline. It can be wrapped
    in a process it doesn't block the UI.

    Args:
        fuzzy_opts (Namespace): Options for fuzzy feature importance.
        fi_opts (Namespace): Options for feature importance.
        ml_opts (MachineLearningOptions): Options for machine learning.
        exec_opts (ExecutionOptions):
        experiment_name (str): The name of the experiment.
    """
    seed = exec_opts.random_state
    set_seed(seed)
    logger_instance = Logger(log_dir(experiment_name))
    logger = logger_instance.make_logger()

    data = DataBuilder(
        data_path=exec_opts.data_path,
        data_split=exec_opts.data_split,
        random_state=exec_opts.random_state,
        normalization=exec_opts.normalization,
        n_bootstraps=ml_opts.n_bootstraps,
        logger=logger,
    ).ingest()

    # Machine learning
    if exec_opts.is_machine_learning:
        trained_models = train.run(ml_opts, data, logger)
        if ml_opts.save_models:
            for model_name in trained_models:
                for i, model in enumerate(trained_models[model_name]):
                    save_path = ml_model_dir(experiment_name) / f"{model_name}-{i}.pkl"
                    save_model(model, save_path)
    else:
        trained_models = load_models(ml_model_dir(experiment_name))

    # Feature importance
    if exec_opts.is_feature_importance:
        gloabl_importance_results, local_importance_results, ensemble_results = (
            feature_importance.run(fi_opts, data, trained_models, logger)
        )

        # Fuzzy interpretation
        if fuzzy_opts.fuzzy_feature_selection:
            fuzzy_rules = fuzzy_interpretation.run(
                fuzzy_opts, ml_opts, data, trained_models, ensemble_results, logger
            )

    # Close the logger
    close_logger(logger_instance, logger)


def cancel_pipeline(p: Process):
    """Cancel a running pipeline.

    Args:
        p (Process): the process running the pipeline to cancel.
    """
    if p.is_alive():
        p.terminate()


## Page contents
st.set_page_config(
    page_title="BioFEFI",
    page_icon=sidebar_logo(),
)
header_logo()
sidebar_logo()
with st.sidebar:
    st.header("Options")
    # st.checkbox("Feature Engineering", key=ConfigStateKeys.IsFeatureEngineering)

    # Machine Learning Options
    ml_options_box()

    # Feature Importance (+ Fuzzy) Options
    fi_options_box()

    # Global plot options
    plot_options_box()
    seed = st.number_input(
        "Random seed", value=1221, min_value=0, key=ConfigStateKeys.RandomSeed
    )
data_upload_form()


# If the user has uploaded a file and pressed the run button, run the pipeline
if (
    uploaded_file := st.session_state.get(ConfigStateKeys.UploadedFileName)
) and st.session_state.get(ExecutionStateKeys.RunPipeline, False):
    experiment_name = st.session_state.get(ConfigStateKeys.ExperimentName)
    upload_path = uploaded_file_path(uploaded_file.name, experiment_name)
    save_upload(upload_path, uploaded_file.read().decode("utf-8"))
    if uploaded_models := st.session_state.get(ConfigStateKeys.UploadedModels):
        for m in uploaded_models:
            upload_path = ml_model_dir(experiment_name) / m.name
            save_upload(upload_path, m.read(), "wb")
    config = build_configuration()
    process = Process(target=pipeline, args=config, daemon=True)
    process.start()
    cancel_button = st.button("Cancel", on_click=cancel_pipeline, args=(process,))
    with st.spinner("Running pipeline..."):
        # wait for the process to finish or be cancelled
        process.join()
    st.session_state[ConfigStateKeys.LogBox] = get_logs(log_dir(experiment_name))
    log_box()
    ml_plots = ml_plot_dir(experiment_name)
    if ml_plots.exists():
        plot_box(ml_plots, "Machine learning plots")
    fi_plots = fi_plot_dir(experiment_name)
    if fi_plots.exists():
        plot_box(fi_plots, "Feature importance plots")
    fuzzy_plots = fuzzy_plot_dir(experiment_name)
    if fuzzy_plots.exists():
        plot_box(fuzzy_plots, "Fuzzy plots")
