import logging
from biofefi.options.enums import ProblemTypes
from biofefi.services.ml_models import get_models


def test_get_models_returns_types():
    # Arrange
    model_types = {
        "Random Forest": {
            "use": True,
            "params": {
                "n_estimators": 100,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_depth": None,
            },
        }
    }
    logger = logging.Logger("test_ml_models")

    # Act
    models = get_models(
        model_types=model_types, problem_type=ProblemTypes.Regression, logger=logger
    )

    # Assert
    for model in models.values():
        assert isinstance(model, type)
