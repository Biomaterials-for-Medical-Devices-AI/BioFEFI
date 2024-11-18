from dataclasses import dataclass

from biofefi.options.enums import Normalisations, ProblemTypes


@dataclass
class ExecutionOptions:
    data_path: str
    data_split: dict | None = None
    experiment_name: str = "test"
    is_feature_engineering: bool = False
    is_machine_learning: bool = False
    is_feature_importance: bool = False
    random_state: int = 1221
    problem_type: ProblemTypes = ProblemTypes.Classification
    dependent_variable: str | None = None
    normalization: Normalisations = Normalisations.NoNormalisation
