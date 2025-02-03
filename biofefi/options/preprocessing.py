from dataclasses import dataclass


@dataclass
class PreprocessingOptions:
    feature_selection_methods: dict
    dependent_variable_normalisation: str = "none"
    independent_variable_transformation: str = "none"
