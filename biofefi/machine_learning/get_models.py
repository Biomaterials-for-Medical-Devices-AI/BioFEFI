from typing import Dict, List
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from biofefi.machine_learning.nn_models import BayesianRegularisedNeuralNets

from biofefi.options.enums import ProblemTypes, ModelNames
from biofefi.utils.utils import assert_model_param

# ---- Mapping of model names and problem types to their corresponding classes ---->

_MODEL_PROBLEM_DICT = {
    (ModelNames.LinearModel, ProblemTypes.Classification): LogisticRegression,
    (ModelNames.LinearModel, ProblemTypes.Regression): LinearRegression,
    (ModelNames.RandomForest, ProblemTypes.Classification): RandomForestClassifier,
    (ModelNames.RandomForest, ProblemTypes.Regression): RandomForestRegressor,
    (ModelNames.XGBoost, ProblemTypes.Classification): XGBClassifier,
    (ModelNames.XGBoost, ProblemTypes.Regression): XGBRegressor,
    (ModelNames.SVM, ProblemTypes.Classification): SVC,
    (ModelNames.SVM, ProblemTypes.Regression): SVR,
    (
        ModelNames.BayesianRegularisedNeuralNets,
        ProblemTypes.Classification,
    ): BayesianRegularisedNeuralNets,
    (
        ModelNames.BayesianRegularisedNeuralNets,
        ProblemTypes.Regression,
    ): BayesianRegularisedNeuralNets,
}


def get_models(
    model_types: Dict[str, Dict], problem_type: str, logger: object = None
) -> List:
    """
    Initialize and return a dictionary of models based on
    specified types and problem domain.

    Parameters
    ----------
    model_types : Dict[str, Dict]
        Dictionary where keys are model names and values are
        dictionaries containing model parameters and
        usage flags.

    problem_type : str
        The type of problem ('classification' or 'regression').

    logger : object, optional
        Logger object for logging information.

    Returns
    -------
    models : Dict[str, object]
        Dictionary of initialized model instances.
    """
    models = {}
    model_list = [
        (model_type, model["params"])
        for model_type, model in model_types.items()
        if model["use"]
    ]
    for model, model_param in model_list:

        model_class = _MODEL_PROBLEM_DICT.get((model, problem_type.lower()))
        if model_class:
            if model == "BayesianRegularisedNeuralNets":

                model_param["problem_type"] = problem_type.lower()
                opt = SimpleNamespace(**model_param)
                models[model] = model_class(opt=opt)

            else:
                if problem_type.lower() == ProblemTypes.Classification:
                    model_param = assert_model_param(
                        model_class, model_param, logger=logger
                    )
                    model_param["class_weight"] = "balanced"
                    models[model] = model_class(**model_param)
                else:
                    model_param = assert_model_param(
                        model_class, model_param, logger=logger
                    )
                    models[model] = model_class(**model_param)

        else:
            raise ValueError(f"Model type {model} not recognized")
    return models
