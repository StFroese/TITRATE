import pytest


def test_AsmyptoticValidator(measurement_dataset, asimov_dataset):
    from titrate.validation import AsymptoticValidator

    validator = AsymptoticValidator(measurement_dataset, asimov_dataset, "qmu", "scale")
    result = validator.validate(n_toys=10)
    assert result["valid"]

    # same for qtildemu
    validator_tilde = AsymptoticValidator(
        measurement_dataset, asimov_dataset, "qtildemu", "scale"
    )
    result_tilde = validator_tilde.validate(n_toys=10)
    assert result_tilde["valid"]

    with pytest.raises(ValueError) as excinfo:
        AsymptoticValidator(measurement_dataset, asimov_dataset, "stupidTest", "scale")

    assert str(excinfo.value) == "Statistic must be one of ['qmu', 'qtildemu']"
