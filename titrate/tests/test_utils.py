import numpy as np


def test_copy_dataset_with_models(measurement_dataset):
    from titrate.utils import copy_dataset_with_models

    measurement_copy = copy_dataset_with_models(measurement_dataset)

    assert np.all((measurement_copy.counts == measurement_dataset.counts).data)
    assert np.all((measurement_copy.npred() == measurement_dataset.npred()).data)

    assert (
        measurement_dataset.models.parameters.names
        == measurement_copy.models.parameters.names
    )
    assert np.all(
        measurement_dataset.models.parameters.value
        == measurement_copy.models.parameters.value
    )


def test_calc_ts_toyMC(measurement_dataset):
    from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic
    from titrate.utils import calc_ts_toyMC

    ts_sample_0 = calc_ts_toyMC(measurement_dataset, QMuTestStatistic, 10, 1, "scale")
    ts_sample_1 = calc_ts_toyMC(measurement_dataset, QMuTestStatistic, 10, 1, "scale")

    assert ts_sample_0 != ts_sample_1

    ts_tilde_sample_0 = calc_ts_toyMC(
        measurement_dataset, QTildeMuTestStatistic, 10, 1, "scale"
    )
    ts_tilde_sample_1 = calc_ts_toyMC(
        measurement_dataset, QTildeMuTestStatistic, 10, 1, "scale"
    )

    assert ts_tilde_sample_0 != ts_tilde_sample_1
