# import third party impots
import pytest
from pydantic import ValidationError

# Internal Imports
from fraud_detection.config.config import ClassWeightsConfig


def test_default_class_weight_config():
    cw = ClassWeightsConfig()
    assert cw.pos_weight == 1.0, "Default value of pos_weight has to be 1.0"
    assert cw.neg_weight == 1.0, "Default value of neg_weight has to be 1.0"


def test_wrong_class_weight_config_raises_validation_error():
    # Only assert exception raised and that the failing field is referenced in the error details.
    with pytest.raises(ValidationError) as exc_info:
        ClassWeightsConfig(pos_weight=-1)

    # Do not assert full error message — check that the field name appears in the repr
    msg = str(exc_info.value)
    assert "pos_weight" in msg or "pos_weight" in repr(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        ClassWeightsConfig(neg_weight=-1)

    # Do not assert full error message — check that the field name appears in the repr
    msg = str(exc_info.value)
    assert "neg_weight" in msg or "neg_weight" in repr(exc_info.value)
