from concurrent.futures import ProcessPoolExecutor

import astropy.units as u
import numpy as np
from astropy.table import QTable
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    SkyModel,
    TemplateSpatialModel,
)
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm

from titrate.datasets import AsimovMapDataset
from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic
from titrate.utils import copy_models_to_dataset

STATISTICS = {"qmu": QMuTestStatistic, "qtildemu": QTildeMuTestStatistic}
CS = DarkMatterAnnihilationSpectralModel.THERMAL_RELIC_CROSS_SECTION


class ULCalculator:
    def __init__(
        self,
        measurement_dataset,
        statistic="qmu",
        poi_name="scale",
        cl=0.95,
        cl_type="s",
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
        while self.pvalue(poi_ul, cl_type=self.cl_type) > 1 - self.cl or np.isnan(
            self.pvalue(poi_ul, cl_type=self.cl_type)
        ):
            poi_ul *= 2

        interp_ul_points = np.linspace(poi_ul / 2, poi_ul, 10)
        interp_pvalues = np.array(
            [
                self.pvalue(interp_ul, cl_type=self.cl_type)
                for interp_ul in interp_ul_points
            ]
        ).ravel()
        interpolation = interp1d(interp_ul_points, interp_pvalues - 1 + self.cl)
        poi_ul = brentq(interpolation, poi_ul / 2, poi_ul)

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


class ULFactory:
    def __init__(
        self,
        measurement_dataset,
        channels,
        mass_min,
        mass_max,
        n_steps,
        jfactor_map,
        cl_type="s",
        cl=0.95,
        max_workers=None,
        **kwargs,
    ):
        self.measurement_dataset = measurement_dataset
        self.channels = channels
        self.masses = np.geomspace(
            mass_min.to_value("TeV"), mass_max.to_value("TeV"), n_steps
        )
        self.jfactor_map = jfactor_map
        self.cl_type = cl_type
        self.cl = cl
        self.kwargs = kwargs
        self.max_workers = max_workers
        self.kwargs["cl"] = self.cl
        self.kwargs["cl_type"] = self.cl_type
        self.uls = None
        self.expected_uls = None

    def setup_models(self):
        models = []
        spatial_model = TemplateSpatialModel(self.jfactor_map, normalize=False)
        bkg_model = FoVBackgroundModel(dataset_name="foo")
        for channel in self.channels:
            for mass in self.masses:
                spectral_model = DarkMatterAnnihilationSpectralModel(
                    mass=mass * u.TeV, channel=channel
                )
                sky_model = SkyModel(
                    spatial_model=spatial_model,
                    spectral_model=spectral_model,
                    name=f"mass_{mass:.2f}TeV_channel_{channel}",
                )
                models.append(Models([sky_model, bkg_model]))

        return models

    def setup_calculator(self, models):
        measurement_copy = self.measurement_dataset.copy()
        copy_models_to_dataset(models, measurement_copy)
        return ULCalculator(measurement_copy, **self.kwargs)

    def compute_uls(self):
        with ProcessPoolExecutor(self.max_workers) as pool:
            futures = [
                pool.submit(self.setup_calculator(models).compute)
                for models in self.setup_models()
            ]
            uls = [future.result() for future in futures]
        return uls

    def compute_expected(self):
        with ProcessPoolExecutor(self.max_workers) as pool:
            futures = [
                pool.submit(self.setup_calculator(models).expected_uls)
                for models in self.setup_models()
            ]
            expected_uls = [future.result() for future in futures]
        return expected_uls

    def compute(self):
        self.uls = self.compute_uls()
        self.expected_uls = self.compute_expected()

    def save_results(self, path, overwrite=False, **kwargs):
        if self.uls is None or self.expected_uls is None:
            raise ValueError("No results computed yet. Run compute() first.")

        # prepare uls
        n_channels = len(self.channels)
        uls = np.array(self.uls).reshape(n_channels, -1)
        median_uls = np.array([ul["med"] for ul in self.expected_uls]).reshape(
            n_channels, -1
        )
        one_sigma_minus_uls = np.array(
            [ul["1sig"][0] for ul in self.expected_uls]
        ).reshape(n_channels, -1)
        one_sigma_plus_uls = np.array(
            [ul["1sig"][1] for ul in self.expected_uls]
        ).reshape(n_channels, -1)
        two_sigma_minus_uls = np.array(
            [ul["2sig"][0] for ul in self.expected_uls]
        ).reshape(n_channels, -1)
        two_sigma_plus_uls = np.array(
            [ul["2sig"][1] for ul in self.expected_uls]
        ).reshape(n_channels, -1)

        for ch_idx, channel in enumerate(self.channels):
            # to dict
            ul_dict = {
                "mass": self.masses * u.TeV,
                "channel": np.repeat(channel, len(self.masses)),
                "cl_type": np.repeat(self.cl_type, len(self.masses)),
                "cl": np.repeat(self.cl, len(self.masses)),
                "ul": uls[ch_idx] * CS,
                "median_ul": median_uls[ch_idx] * CS,
                "1sigma_minus_ul": one_sigma_minus_uls[ch_idx] * CS,
                "1sigma_plus_ul": one_sigma_plus_uls[ch_idx] * CS,
                "2sigma_minus_ul": two_sigma_minus_uls[ch_idx] * CS,
                "2sigma_plus_ul": two_sigma_plus_uls[ch_idx] * CS,
            }
            qtable = QTable(ul_dict)
            qtable.write(
                path,
                format="hdf5",
                path=f"upperlimits/{channel}",
                overwrite=overwrite,
                append=True,
                serialize_meta=True,
                **kwargs,
            )
