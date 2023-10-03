import astropy.units as u
import numpy as np
import pytest


@pytest.fixture(scope="module")
def validation_file(measurement_dataset, tmp_path_factory):
    from titrate.validation import AsymptoticValidator

    data = tmp_path_factory.mktemp("data")

    validator = AsymptoticValidator(measurement_dataset, "qmu", "scale")
    result = validator.validate(n_toys=10)
    assert list(result.keys()) == ["pvalue_diff", "pvalue_same", "valid"]
    assert result["pvalue_diff"] != 0
    assert result["pvalue_diff"] != np.nan
    assert result["pvalue_same"] != 0
    assert result["pvalue_same"] != np.nan
    assert isinstance(result["valid"], np.bool_)

    with pytest.raises(ValueError) as excinfo:
        AsymptoticValidator(measurement_dataset, "stupidTest", "scale")

    assert str(excinfo.value) == "Statistic must be one of ['qmu', 'qtildemu']"

    validator.write(f"{data}/val.h5")

    validator_tilde = AsymptoticValidator(measurement_dataset, "qtildemu", "scale")
    result_tilde = validator_tilde.validate(n_toys=10)
    assert list(result.keys()) == ["pvalue_diff", "pvalue_same", "valid"]
    assert result_tilde["pvalue_diff"] != 0
    assert result_tilde["pvalue_diff"] != np.nan
    assert result_tilde["pvalue_same"] != 0
    assert result_tilde["pvalue_same"] != np.nan
    assert isinstance(result_tilde["valid"], np.bool_)

    validator_tilde.write(f"{data}/val.h5")

    return f"{data}/val.h5"


@pytest.mark.parametrize("statistic", ["qmu", "qtildemu"])
def test_AsmyptoticValidator(measurement_dataset, statistic, validation_file):
    from titrate.validation import AsymptoticValidator

    validator = AsymptoticValidator(
        measurement_dataset,
        statistic=statistic,
        path=validation_file,
        channel="b",
        mass=50 * u.TeV,
    )
    result = validator.validate()

    assert list(result.keys()) == ["pvalue_diff", "pvalue_same", "valid"]
    assert result["pvalue_diff"] != 0
    assert result["pvalue_diff"] != np.nan
    assert result["pvalue_same"] != 0
    assert result["pvalue_same"] != np.nan
    assert isinstance(result["valid"], np.bool_)


@pytest.mark.parametrize("statistic", ["qmu", "qtildemu"])
def test_ValidationPlotter(measurement_dataset, statistic, validation_file):
    from titrate.plotting import ValidationPlotter

    ValidationPlotter(
        measurement_dataset,
        path=validation_file,
        statistic=statistic,
        channel="b",
        mass=50 * u.TeV,
    )
