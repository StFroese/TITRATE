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


def test_pdf_approximations(asimov_dataset):
    qmu = QMuTestStatistic(asimov_dataset, poi_name="scale")

    area = quad(
        lambda x: qmu.asympotic_approximation_pdf(x, poi_val=1, poi_true_val=1),
        0,
        np.inf,
    )[0]
    assert approx(area) == 0.5

    area = quad(
        lambda x: qmu.asympotic_approximation_pdf(x, poi_val=1, poi_true_val=0),
        0,
        np.inf,
    )[0]
    assert approx(area) == 1 - norm.cdf(
        -1 / qmu.sigma()
    )  # given by asymptotic formula from paper

    # Same for QTildeMuTestStatistic
    qtildemu = QTildeMuTestStatistic(asimov_dataset, poi_name="scale")

    area = quad(
        lambda x: qtildemu.asympotic_approximation_pdf(x, poi_val=1, poi_true_val=1),
        0,
        np.inf,
    )[0]
    assert approx(area) == 0.5

    area = quad(
        lambda x: qtildemu.asympotic_approximation_pdf(x, poi_val=1, poi_true_val=0),
        0,
        np.inf,
    )[0]
    assert approx(area) == 1 - norm.cdf(
        -1 / qtildemu.sigma()
    )  # given by asymptotic formula from paper


def test_cdf_approximations(asimov_dataset):
    qmu = QMuTestStatistic(asimov_dataset, poi_name="scale")

    area = quad(
        lambda x: qmu.asympotic_approximation_pdf(x, poi_val=1, poi_true_val=1), 0, 5
    )[0]
    assert approx(area + 0.5) == qmu.asympotic_approximation_cdf(
        5, poi_val=1, poi_true_val=1
    )

    area = quad(
        lambda x: qmu.asympotic_approximation_pdf(x, poi_val=1, poi_true_val=0), 0, 5
    )[0]
    assert approx(
        area + norm.cdf(-1 / qmu.sigma()), rel=1e-3
    ) == qmu.asympotic_approximation_cdf(5, poi_val=1, poi_true_val=0)

    # Same for QTildeMuTestStatistic
    qtildemu = QTildeMuTestStatistic(asimov_dataset, poi_name="scale")

    area = quad(
        lambda x: qtildemu.asympotic_approximation_pdf(x, poi_val=1, poi_true_val=1),
        0,
        5,
    )[0]
    assert approx(area + 0.5) == qtildemu.asympotic_approximation_cdf(
        5, poi_val=1, poi_true_val=1
    )

    area = quad(
        lambda x: qtildemu.asympotic_approximation_pdf(x, poi_val=1, poi_true_val=0),
        0,
        5,
    )[0]
    assert approx(
        area + norm.cdf(-1 / qmu.sigma()), rel=1e-3
    ) == qtildemu.asympotic_approximation_cdf(5, poi_val=1, poi_true_val=0)


def test_pvalue(asimov_dataset):
    qmu = QMuTestStatistic(asimov_dataset, poi_name="scale")

    assert approx(qmu.pvalue(0, 1, 1)) == 0.5
    assert approx(qmu.pvalue(np.inf, 1, 1)) == 0
    assert approx(qmu.pvalue(0, 1, 0)) == 1 - norm.cdf(-1 / qmu.sigma())
    assert approx(qmu.pvalue(np.inf, 1, 0)) == 0

    # Same for QTildeMuTestStatistic
    qtildemu = QTildeMuTestStatistic(asimov_dataset, poi_name="scale")

    assert approx(qtildemu.pvalue(0, 1, 1)) == 0.5
    assert approx(qtildemu.pvalue(np.inf, 1, 1)) == 0
    assert approx(qtildemu.pvalue(0, 1, 0)) == 1 - norm.cdf(-1 / qtildemu.sigma())
    assert approx(qtildemu.pvalue(np.inf, 1, 0)) == 0


def test_significance(asimov_dataset):
    qmu = QMuTestStatistic(asimov_dataset, poi_name="scale")

    assert approx(qmu.significance(25, 1, 1)) == 5
    assert approx(qmu.significance(0, 1, 0)) == -1 / qmu.sigma()

    qtildemu = QTildeMuTestStatistic(asimov_dataset, poi_name="scale")

    assert approx(qtildemu.significance(25, 0, 0)) == np.inf
    assert approx(qtildemu.significance(25, 1, 1)) == (
        25 + 1 / qtildemu.sigma() ** 2
    ) / (2 / qtildemu.sigma())
    assert approx(qtildemu.significance(0.1, 1, 1)) == np.sqrt(0.1)
    assert approx(qtildemu.significance(0, 1, 0)) == -1 / qtildemu.sigma()
