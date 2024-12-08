from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from biofefi.machine_learning.nn_models import BayesianRegularisedNN
from biofefi.options.enums import ModelNames, ProblemTypes
from biofefi.utils.utils import assert_model_param

_MODEL_PROBLEM_DICT = {
    (ModelNames.LinearModel, ProblemTypes.Classification): LogisticRegression,
    (ModelNames.LinearModel, ProblemTypes.Regression): LinearRegression,
    (ModelNames.RandomForest, ProblemTypes.Classification): RandomForestClassifier,
    (ModelNames.RandomForest, ProblemTypes.Regression): RandomForestRegressor,
    (ModelNames.XGBoost, ProblemTypes.Classification): XGBClassifier,
    (ModelNames.XGBoost, ProblemTypes.Regression): XGBRegressor,
    (ModelNames.SVM, ProblemTypes.Classification): SVC,
    (ModelNames.SVM, ProblemTypes.Regression): SVR,
    (ModelNames.BRNN, ProblemTypes.Classification): BayesianRegularisedNN,
    (ModelNames.BRNN, ProblemTypes.Regression): BayesianRegularisedNN,
}


def get_models(
    model_types: Dict[str, Dict], problem_type: str, logger: object = None
) -> List:
    models = {}
    model_list = [
        (model_type, model["params"])
        for model_type, model in model_types.items()
        if model["use"]
    ]
    for model, model_param in model_list:
        if model_class := _MODEL_PROBLEM_DICT.get(
            (model.lower(), problem_type.lower())
        ):
            if model == ModelNames.BRNN:
                models[model] = model_class(problem_type=problem_type, **model_param)

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
