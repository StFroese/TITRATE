import numpy as np
from gammapy.modeling import Fit
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm

from titrate.datasets import AsimovMapDataset
from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic

STATISTICS = {"qmu": QMuTestStatistic, "qtildemu": QTildeMuTestStatistic}


class ULCalculator:
    def __init__(
        self, measurement_dataset, statistic="qmu", poi_name="", cl=0.95, cl_type="s"
    ):
        self.measurement_dataset = measurement_dataset
        self.poi_name = poi_name
        if statistic not in STATISTICS.keys():
            raise ValueError(
                "Statistic must be one of {}".format(list(STATISTICS.keys()))
            )
        self.statistic = STATISTICS[statistic](
            self.measurement_dataset, poi_name=self.poi_name
        )

        if cl_type not in ["s+b", "s"]:
            raise ValueError('cl_type must be either "s+b" or "s"')
        self.cl_type = cl_type
        self.cl = cl

    def compute(self):
        poi_best = self.statistic.poi_best

        if poi_best < 0:
            poi_ul = 1e-2
        else:
            poi_ul = poi_best + 0.01 * poi_best
        while self.pvalue(poi_ul, cl_type=self.cl_type) > 1 - self.cl:
            poi_ul *= 2

        interp_ul_points = np.linspace(poi_ul / 10, poi_ul, 10)
        interp_pvalues = np.array(
            [
                self.pvalue(interp_ul, cl_type=self.cl_type)
                for interp_ul in interp_ul_points
            ]
        ).ravel()
        interpolation = interp1d(interp_ul_points, interp_pvalues - 1 + self.cl)
        poi_ul = brentq(interpolation, poi_ul / 10, poi_ul)

        return poi_ul

    def pvalue(self, poi_ul, cl_type):
        pval_bkg = 0
        if self.statistic.__class__.__name__ == "QTildeMuTestStatistic":
            asimov_dataset = AsimovMapDataset.from_MapDataset(self.measurement_dataset)
            asimov_dataset.models.parameters[self.poi_name].value = poi_ul
            asimov_dataset.fake()
            statistic = self.statistic.__class__(asimov_dataset, poi_name=self.poi_name)
            ts_val = self.statistic.evaluate(poi_ul)  # ts_val on measurement_dataset

            pval_sig_bkg = statistic.pvalue(poi_ul, ts_val=ts_val)

            if cl_type == "s":
                no_signal_asimov_dataset = AsimovMapDataset.from_MapDataset(
                    self.measurement_dataset
                )
                no_signal_asimov_dataset.models.parameters[self.poi_name].value = 0
                no_signal_asimov_dataset.fake()
                no_signal_statistic = self.statistic.__class__(
                    no_signal_asimov_dataset, poi_name=self.poi_name
                )
                ts_val_no_signal = self.statistic.evaluate(
                    0
                )  # ts_val on measurement_dataset with no signal

                pval_bkg = no_signal_statistic.pvalue(0, ts_val=ts_val_no_signal)

        else:
            pval_sig_bkg = self.statistic.pvalue(poi_ul)

            if cl_type == "s":
                pval_bkg = self.statistic.pvalue(0)

        return pval_sig_bkg / (1 - pval_bkg)

    def expected_uls(self):
        # Create asimov dataset
        asimov_dataset = AsimovMapDataset.from_MapDataset(self.measurement_dataset)
        asimov_dataset.models.parameters[self.poi_name].value = 0
        asimov_dataset.fake()

        fit = Fit()
        fit_result = fit.run(datasets=[asimov_dataset])

        sigma = np.sqrt(fit_result.covariance_result.matrix[0, 0])
        med_ul = self.compute_band(sigma, 0, self.cl_type)

        one_sig_plus = self.compute_band(sigma, 1, self.cl_type)
        one_sig_minus = self.compute_band(sigma, -1, self.cl_type)

        two_sig_plus = self.compute_band(sigma, 2, self.cl_type)
        two_sig_minus = self.compute_band(sigma, -2, self.cl_type)

        return {
            "med": med_ul,
            "1sig": [one_sig_minus, one_sig_plus],
            "2sig": [two_sig_minus, two_sig_plus],
        }

    def compute_band(self, sigma, n_sigma, cl_type):
        if cl_type == "s+b":
            return sigma * (norm.ppf(self.cl) + n_sigma)
        elif cl_type == "s":
            return sigma * (norm.ppf(1 - (1 - self.cl) * norm.pdf(n_sigma)) + n_sigma)
