import numpy as np
from pytest import approx
from scipy.integrate import quad
from scipy.stats import norm

from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic


def test_QMuTestStatistic(measurement_dataset):
    qmu = QMuTestStatistic(measurement_dataset, poi_name="scale")

    assert qmu.check_for_pois() == ["scale", "norm"]
    assert qmu.evaluate(-10) == np.array([0])
    assert qmu.evaluate(10) != np.array([0])


def test_QTildeMuTestStatistic(measurement_dataset, nosignal_dataset):
    qtildemu = QTildeMuTestStatistic(measurement_dataset, poi_name="scale")

    assert qtildemu.check_for_pois() == ["scale", "norm"]
    assert qtildemu.evaluate(-10) == np.array([0])
    assert qtildemu.evaluate(10) != np.array([0])

    # Force negative poi_best by using less counts
    qtildemu = QTildeMuTestStatistic(nosignal_dataset, poi_name="scale")
    assert qtildemu.poi_best < 0
    assert qtildemu.evaluate(qtildemu.poi_best + 10) != np.array([0])


def test_approximations(asimov_dataset):
    qmu = QMuTestStatistic(asimov_dataset, poi_name="scale")

    area = quad(
        lambda x: qmu.asympotic_approximation(x, poi_val=1, poi_true_val=1), 0, np.inf
    )[0]
    assert approx(area) == 0.5

    area = quad(
        lambda x: qmu.asympotic_approximation(x, poi_val=1, poi_true_val=0), 0, np.inf
    )[0]
    assert approx(area) == 1 - norm.cdf(
        -1 / qmu.sigma()
    )  # given by asymptotic formula from paper

    # Same for QTildeMuTestStatistic
    qtildemu = QTildeMuTestStatistic(asimov_dataset, poi_name="scale")

    area = quad(
        lambda x: qtildemu.asympotic_approximation(x, poi_val=1, poi_true_val=1),
        0,
        np.inf,
    )[0]
    assert approx(area) == 0.5

    area = quad(
        lambda x: qtildemu.asympotic_approximation(x, poi_val=1, poi_true_val=0),
        0,
        np.inf,
    )[0]
    assert approx(area) == 1 - norm.cdf(
        -1 / qtildemu.sigma()
    )  # given by asymptotic formula from paper
