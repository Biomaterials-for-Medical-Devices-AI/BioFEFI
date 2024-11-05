from dataclasses import dataclass


@dataclass
class ExecutionOptions:
    data_path: str
    data_split: dict | None = None
    experiment_name: str = "test"
    is_feature_engineering: bool = False
    is_machine_learning: bool = False
    is_feature_importance: bool = False
    random_state: int = 1221
    problem_type: str = "classification"
    dependent_variable: str | None = None
    normalization: str = "None"
