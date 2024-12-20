from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from biofefi.machine_learning.nn_models import (
    BayesianRegularisedNNClassifier,
    BayesianRegularisedNNRegressor,
)
from biofefi.options.enums import (
    DataSplitMethods,
    ModelNames,
    Normalisations,
    ProblemTypes,
    SvmKernels,
)

SVM_KERNELS = [
    SvmKernels.RBF.upper(),  # appear as RBF, not Rbf
    SvmKernels.Linear.capitalize(),
    SvmKernels.Poly.capitalize(),
    SvmKernels.Sigmoid.capitalize(),
    SvmKernels.Precomputed.capitalize(),
]
PROBLEM_TYPES = [
    ProblemTypes.Classification.capitalize(),
    ProblemTypes.Regression.capitalize(),
]
NORMALISATIONS = [
    Normalisations.Standardization.capitalize(),
    Normalisations.MinMax.capitalize(),
    Normalisations.NoNormalisation.capitalize(),
]
PLOT_FONT_FAMILIES = ["serif", "sans-serif", "cursive", "fantasy", "monospace"]
DATA_SPLITS = [
    DataSplitMethods.Holdout.capitalize(),
    DataSplitMethods.KFold.capitalize(),
]
MODEL_PROBLEM_CHOICES = {
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
