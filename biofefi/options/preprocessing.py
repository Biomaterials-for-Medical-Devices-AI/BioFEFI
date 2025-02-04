from dataclasses import dataclass


@dataclass
class PreprocessingOptions:
    feature_selection_methods: dict
    independent_variable_normalisation: str = "none"
    dependent_variable_transformation: str = "none"
