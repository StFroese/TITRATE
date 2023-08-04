import numpy as np


def test_asimov_dataset(asimov_dataset):
    assert np.all((asimov_dataset.counts == asimov_dataset.npred()).data)


def test_measurement_dataset(measurement_dataset):
    assert not np.all((measurement_dataset.counts == measurement_dataset.npred()).data)


def test_compare_asimov_measurement_datasets(asimov_dataset, measurement_dataset):
    assert np.all((measurement_dataset.npred() == measurement_dataset.npred()).data)
