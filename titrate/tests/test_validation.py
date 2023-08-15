import numpy as np
import pytest


def test_AsmyptoticValidator(measurement_dataset, asimov_dataset):
    from titrate.validation import AsymptoticValidator

    validator = AsymptoticValidator(measurement_dataset, asimov_dataset, "qmu", "scale")
    result = validator.validate(n_toys=10)
    assert list(result.keys()) == ["pvalue_diff", "pvalue_same", "valid"]
    assert result["pvalue_diff"] != 0
    assert result["pvalue_diff"] != np.nan
    assert result["pvalue_same"] != 0
    assert result["pvalue_same"] != np.nan
    assert isinstance(result["valid"], np.bool_)

    # same for qtildemu
    validator_tilde = AsymptoticValidator(
        measurement_dataset, asimov_dataset, "qtildemu", "scale"
    )
    result_tilde = validator_tilde.validate(n_toys=10)
    assert list(result_tilde.keys()) == ["pvalue_diff", "pvalue_same", "valid"]
    assert result_tilde["pvalue_diff"] != 0
    assert result_tilde["pvalue_diff"] != np.nan
    assert result_tilde["pvalue_same"] != 0
    assert result_tilde["pvalue_same"] != np.nan
    assert isinstance(result_tilde["valid"], np.bool_)

    with pytest.raises(ValueError) as excinfo:
        AsymptoticValidator(measurement_dataset, asimov_dataset, "stupidTest", "scale")

    assert str(excinfo.value) == "Statistic must be one of ['qmu', 'qtildemu']"
