from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from biofefi.machine_learning.nn_models import (
    BayesianRegularisedNNClassifier,
    BayesianRegularisedNNRegressor,
)
from biofefi.options.enums import ModelNames, ProblemTypes


MODEL_PROBLEM_DICT = {
    (ModelNames.LinearModel, ProblemTypes.Classification): LogisticRegression,
    (ModelNames.LinearModel, ProblemTypes.Regression): LinearRegression,
    (ModelNames.RandomForest, ProblemTypes.Classification): RandomForestClassifier,
    (ModelNames.RandomForest, ProblemTypes.Regression): RandomForestRegressor,
    (ModelNames.XGBoost, ProblemTypes.Classification): XGBClassifier,
    (ModelNames.XGBoost, ProblemTypes.Regression): XGBRegressor,
    (ModelNames.SVM, ProblemTypes.Classification): SVC,
    (ModelNames.SVM, ProblemTypes.Regression): SVR,
    (
        ModelNames.BRNNClassifier,
        ProblemTypes.Classification,
    ): BayesianRegularisedNNClassifier,
    (ModelNames.BRNNRegressor, ProblemTypes.Regression): BayesianRegularisedNNRegressor,
}
