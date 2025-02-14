import json
import os
from pathlib import Path
from pickle import UnpicklingError, dump, load

from biofefi.options.choices.ml_models import CLASSIFIERS, REGRESSORS
from biofefi.options.enums import ProblemTypes
from biofefi.utils.utils import create_directory


def save_models_metrics(metrics: dict, path: Path):
    """Save the statistical metrics of the models to the given file path.

    Args:
        metrics (dict): The metrics to save.
        path (Path): The file path to save the metrics.
    """

    create_directory(path.parent)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_model(model, path: Path):
    """Save a machine learning model to the given file path.

    Args:
        model (_type_): The model to save. Must be picklable.
        path (Path): The file path to save the model.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "wb") as f:
        dump(model, f, protocol=5)


def load_models(path: Path) -> dict[str, list]:
    """Load pre-trained machine learning models.

    Args:
        path (Path): The path to the directory where the models are saved.

    Returns:
        dict[str, list]: The pre-trained models.
    """
    models: dict[str, list] = dict()
    for file_name in path.iterdir():
        try:
            with open(file_name, "rb") as file:
                model = load(file)
                model_name = model.__class__.__name__
                if model_name in models:
                    models[model_name].append(model)
                else:
                    models[model_name] = [model]
        except UnpicklingError:
            pass  # ignore bad files

    return models


def load_models_to_explain(path: Path, model_names: list) -> dict[str, list]:
    """Load pre-trained machine learning models.

    Args:
        path (Path): The path to the directory where the models are saved.
        model_names (str): The name of the models to explain.

    Returns:
        dict[str, list]: The pre-trained models.
    """
    models: dict[str, list] = dict()
    for file_name in path.iterdir():
        if os.path.basename(file_name) in model_names or model_names == "all":
            try:
                with open(file_name, "rb") as file:
                    model = load(file)
                    model_name = model.__class__.__name__
                    if model_name in models:
                        models[model_name].append(model)
                    else:
                        models[model_name] = [model]
            except UnpicklingError:
                pass  # ignore bad files
    return models


def get_model_type(model_type: str, problem_type: ProblemTypes) -> type:
    """
    Fetch the appropriate type for a given model name based on the problem type.

    Args:
        model_type (dict): The kind of model.
        problem_type (ProblemTypes): Type of problem (classification or regression).

    Raises:
        ValueError: If a model type is not recognised or unsupported.

    Returns:
        type: The constructor for a machine learning model class.
    """
    if problem_type.lower() == ProblemTypes.Classification:
        model_class = CLASSIFIERS.get(model_type.lower())
    elif problem_type.lower() == ProblemTypes.Regression:
        model_class = REGRESSORS.get(model_type.lower())
    if not model_class:
        raise ValueError(f"Model type {model_type} not recognised")

    return model_class


def models_exist(path: Path) -> bool:
    try:
        trained_models = load_models(path)

        if trained_models:
            return True
        else:
            return False

    except Exception:
        return False


# if problem_type.lower() == ProblemTypes.Classification:
#     model_class = CLASSIFIERS.get(model.lower())
#     model_params["class_weight"] = (
#         ["balanced"] if use_grid_search else "balanced"
#     )
# elif problem_type.lower() == ProblemTypes.Regression:
#     model_class = REGRESSORS.get(model.lower())

# models[model] = model_class(**model_params) if use_params else model_class()
# logger.info(
#     f"Using model {model_class.__name__} with parameters {model_params}"
# )
# if not model_class:
#     raise ValueError(f"Model type {model} not recognized")
