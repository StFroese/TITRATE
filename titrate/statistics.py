import abc

import numpy as np
from gammapy.modeling import Fit
from scipy.stats import kstwo, norm

from titrate.datasets import AsimovMapDataset, AsimovSpectralDataset


class POIError(IndexError):
    """Parameter of interest is not defined in model"""


class AsimovApproximationError(IndexError):
    """Approimation on AsimovMapDataset fails"""


class TestStatistic(abc.ABC):
    r"""TestStatistic abstract base class.

    This class will build the base class for other test statistics like $q_\mu$
    or $\tilde{q}_\mu$.

    It contains two basic methods:
    1. `evaluate`, which takes a dataset with models to calculate the test statistic.
    2. `asymptotic_approximation`, which evaluates the distribution of the
        test statistic for a given value of the test statistic.
    """

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def asympotic_approximation_pdf(self):
        pass

    @abc.abstractmethod
    def asympotic_approximation_cdf(self):
        pass

    @abc.abstractmethod
    def pvalue(self):
        pass


class QMuTestStatistic(TestStatistic):
    def __init__(self, dataset, poi_name=""):
        self.dataset = dataset

        self.poi_name = poi_name
        try:
            self.dataset.models.parameters[self.poi_name]
        except IndexError:
            raise POIError(
                f"The parameter of interest `{self.poi_name}` is not available. "
                f"Please choose one from the following list: {self.check_for_pois()}"
            )

        self.fit = Fit()
        self.fit_result = self.fit.run(datasets=self.dataset)
        self.poi_best = self.fit_result.parameters[self.poi_name].value
        self.likelihood_minimum = self.dataset.stat_sum()

    def evaluate(self, poi_val):
        """
        Computes the test statistic for a given dataset
        and parameter of interest (POI).

        Parameters
        ----------
        poi_val : float

        Returns
        -------
        ts : float
        global_fit_valid: bool
            True if the global fit is valid, False otherwise.
        """

        global_fit_valid = True
        if self.poi_best > poi_val:
            return np.array([0]), global_fit_valid

        self.dataset.models.parameters[self.poi_name].scan_values = [poi_val]
        stats = self.fit.stat_profile(self.dataset, self.poi_name, reoptimize=True)
        ts = stats["stat_scan"] - self.likelihood_minimum

        # catch the case when the test statistic is negative
        # happens when the best fit value of the POI is not the global minimum
        # of the likelihood
        if ts < 0:
            global_fit_valid = False

        return ts, global_fit_valid

    def check_for_pois(self):
        """POI must be a norm parameter by definition since
        we want to calculate TS for strength parameter of the signal.
        """
        pois = []
        for parameter in self.dataset.models.parameters:
            if parameter.is_norm:
                pois.append(parameter.name)
        return pois

    def sigma(self):
        return np.sqrt(self.fit_result.covariance_result.matrix[0, 0])

    def asympotic_approximation_pdf(
        self, ts_val, poi_val, same=True, poi_true_val=None
    ):
        if same:
            return (
                1
                / (2 * np.sqrt(2 * np.pi * ts_val))
                * np.exp(-0.5 * (np.sqrt(ts_val)) ** 2)
            )

        return (
            1
            / (2 * np.sqrt(2 * np.pi * ts_val))
            * np.exp(
                -0.5 * (np.sqrt(ts_val) - (poi_val - poi_true_val) / self.sigma()) ** 2
            )
        )

    def asympotic_approximation_cdf(
        self, ts_val, poi_val, same=True, poi_true_val=None
    ):
        if same:
            return norm.cdf(np.sqrt(ts_val))

        return norm.cdf(np.sqrt(ts_val) - (poi_val - poi_true_val) / self.sigma())

    def pvalue(self, poi_val, same=True, poi_true_val=None, ts_val=None):
        if ts_val is None:
            ts_val, valid = self.evaluate(poi_val)
            ts_val = ts_val if valid else 0
        if same:
            return 1 - self.asympotic_approximation_cdf(ts_val, poi_val)
        return 1 - self.asympotic_approximation_cdf(
            ts_val, poi_val, same=False, poi_true_val=poi_true_val
        )

    def significance(self, poi_val, same=True, poi_true_val=None, ts_val=None):
        if ts_val is None:
            ts_val, valid = self.evaluate(poi_val)
            ts_val = ts_val if valid else 0
        if same:
            return np.sqrt(ts_val)

        return norm.ppf(
            1
            - self.pvalue(poi_val, same=False, poi_true_val=poi_true_val, ts_val=ts_val)
        )


class QTildeMuTestStatistic(TestStatistic):
    def __init__(self, dataset, poi_name=""):
        self.dataset = dataset

        self.poi_name = poi_name
        try:
            self.dataset.models.parameters[self.poi_name]
        except IndexError:
            raise POIError(
                f"The parameter of interest `{self.poi_name}` is not available. "
                f"Please choose one from the following list: {self.check_for_pois()}"
            )

        self.fit = Fit()
        self.fit_result = self.fit.run(datasets=self.dataset)
        self.poi_best = self.fit_result.parameters[self.poi_name].value
        if self.poi_best < 0 and not (
            isinstance(self.dataset, AsimovMapDataset)
            or isinstance(self.dataset, AsimovSpectralDataset)
        ):
            self.dataset.models.parameters[self.poi_name].value = 0
            self.dataset.models.parameters[self.poi_name].frozen = True
            self.fit_result = self.fit.run(datasets=self.dataset)
            self.dataset.models.parameters[self.poi_name].frozen = False

        self.likelihood_constant = self.dataset.stat_sum()

    def evaluate(self, poi_val):
        """
        Computes the test statistic for a given dataset
        and parameter of interest (POI).

        Parameters
        ----------
        poi_val : float

        Returns
        -------
        ts : float
        global_fit_valid: bool
            True if the global fit is valid, False otherwise.
        """
        if self.poi_best > poi_val:
            return np.array([0])

        self.dataset.models.parameters[self.poi_name].scan_values = [poi_val]
        stats = self.fit.stat_profile(self.dataset, self.poi_name, reoptimize=True)
        ts = stats["stat_scan"] - self.likelihood_constant

        if ts < 0 and np.isclose(ts, 0, atol=1e-03):
            ts = np.array([0])

        return ts

    def check_for_pois(self):
        """POI must be a norm parameter by definition since
        we want to calculate TS for strength parameter of the signal.
        """
        pois = []
        for parameter in self.dataset.models.parameters:
            if parameter.is_norm:
                pois.append(parameter.name)
        return pois

    def sigma(self, poi_val, poi_true_val, same=False):
        if poi_val == 0:
            return np.sqrt(self.fit_result.covariance_result.matrix[0, 0])
        if same:
            return 0

        ts = self.evaluate(poi_val)
        return (poi_val - poi_true_val) / np.sqrt(ts)

    def asympotic_approximation_pdf(
        self, ts_val, poi_val, same=True, poi_true_val=None
    ):
        nc = self.evaluate(poi_val)

        if same:
            sigma = np.sqrt(self.fit_result.covariance_result.matrix[0, 0])
            mu_sigma = poi_val**2 / sigma**2
            return np.where(
                (ts_val > 0) & (ts_val <= mu_sigma),
                (
                    1
                    / (2 * np.sqrt(2 * np.pi) * np.sqrt(ts_val))
                    * np.exp(-0.5 * (np.sqrt(ts_val)) ** 2)
                ),
                (
                    1
                    / (np.sqrt(2 * np.pi) * (2 * np.sqrt(mu_sigma)))
                    * np.exp(
                        -0.5 * (ts_val + mu_sigma) ** 2 / ((2 * np.sqrt(mu_sigma)) ** 2)
                    )
                ),
            )

        sigma = poi_val / np.sqrt(nc)
        mu_sigma = poi_val**2 / sigma**2
        return np.where(
            (ts_val > 0) & (ts_val <= mu_sigma),
            (
                1
                / (2 * np.sqrt(2 * np.pi) * np.sqrt(ts_val))
                * np.exp(-0.5 * (np.sqrt(ts_val) - np.sqrt(nc)) ** 2)
            ),
            (
                1
                / (np.sqrt(2 * np.pi) * (2 * np.sqrt(mu_sigma)))
                * np.exp(-0.5 * (ts_val - nc) ** 2 / ((2 * np.sqrt(mu_sigma)) ** 2))
            ),
        )

    def asympotic_approximation_cdf(
        self, ts_val, poi_val, same=True, poi_true_val=None
    ):
        nc = self.evaluate(poi_val)

        if same:
            sigma = np.sqrt(self.fit_result.covariance_result.matrix[0, 0])
            mu_sigma = poi_val**2 / sigma**2
            return np.where(
                (ts_val > 0) & (ts_val <= mu_sigma),
                norm.cdf(np.sqrt(ts_val)),
                norm.cdf((ts_val + mu_sigma) / (2 * np.sqrt(mu_sigma))),
            )

        sigma = poi_val / np.sqrt(nc)
        mu_sigma = poi_val**2 / sigma**2
        return np.where(
            (ts_val > 0) & (ts_val <= mu_sigma),
            norm.cdf(np.sqrt(ts_val) - np.sqrt(nc)),
            norm.cdf((ts_val - nc) / (2 * np.sqrt(mu_sigma))),
        )

    def pvalue(self, poi_val, same=True, poi_true_val=None, ts_val=None):
        if ts_val is None:
            ts_val = self.evaluate(poi_val)
        if same:
            return 1 - self.asympotic_approximation_cdf(ts_val, poi_val)
        return 1 - self.asympotic_approximation_cdf(
            ts_val, poi_val, same=False, poi_true_val=poi_true_val
        )

    def significance(self, poi_val, same=True, poi_true_val=None, ts_val=None):
        if ts_val is None:
            ts_val, valid = self.evaluate(poi_val)
            ts_val = ts_val if valid else 0
        if same:
            sigma = self.sigma(poi_val, poi_true_val)
            if ts_val > poi_val**2 / sigma**2:
                return (ts_val + poi_val**2 / sigma**2) / (2 * poi_val / sigma)
            else:
                return np.sqrt(ts_val)

        return norm.ppf(
            1
            - self.pvalue(poi_val, same=False, poi_true_val=poi_true_val, ts_val=ts_val)
        )


def kstest(rvs, cdf):
    """Kolmogorov-Smirnov test for goodness of fit."""
    data_cdf = np.sum(rvs[:, None] <= rvs, axis=0) / len(rvs)

    D = np.max(np.abs(cdf(rvs) - data_cdf))

    return kstwo.sf(D, len(rvs))
