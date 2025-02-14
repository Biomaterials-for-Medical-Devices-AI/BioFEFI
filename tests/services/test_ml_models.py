import pytest
from biofefi.options.enums import ProblemTypes
from biofefi.services.ml_models import get_model_type


def test_get_model_type_returns_type():
    # Arrange
    model_type = "Random Forest"

    # Act
    model = get_model_type(model_type, ProblemTypes.Regression)

    # Assert
    assert isinstance(model, type)


def test_get_model_type_throws_value_error():
    # Arrange
    model_types = "Unknown"

    # Act/Assert
    with pytest.raises(ValueError):
        get_model_type(model_types, ProblemTypes.Regression)
