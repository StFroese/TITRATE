from concurrent.futures import ProcessPoolExecutor

import astropy.units as u
import numpy as np
from astropy.table import QTable
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel
from gammapy.modeling import Covariance, Fit
from gammapy.modeling.fit import OptimizeResult
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    SkyModel,
    TemplateSpatialModel,
)
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm

from titrate.datasets import AsimovMapDataset, AsimovSpectralDataset
from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic
from titrate.utils import copy_dataset_with_models, copy_models_to_dataset

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
        analysis="3d",
        mu_guess=None,
    ):
        self.measurement_dataset = measurement_dataset
        self.poi_name = poi_name
        self.stat_class_name = statistic
        if statistic not in STATISTICS.keys():
            raise ValueError(
                "Statistic must be one of {}".format(list(STATISTICS.keys()))
            )
        self.statistic = STATISTICS[statistic](
            copy_dataset_with_models(self.measurement_dataset), poi_name=self.poi_name
        )

        if cl_type not in ["s+b", "s"]:
            raise ValueError('cl_type must be either "s+b" or "s"')
        self.cl_type = cl_type
        self.cl = cl
        self.analysis = analysis
        self.mu_guess = mu_guess

        # probaably need this dataset anyways
        self.d_no_bkg = copy_dataset_with_models(self.measurement_dataset)
        self.d_no_bkg.models.parameters[self.poi_name].value = 0
        self.d_no_bkg.models.parameters[self.poi_name].frozen = True
        fit = Fit()
        _ = fit.run(self.d_no_bkg)
        self.d_no_bkg.models.parameters[self.poi_name].frozen = False
        if self.analysis == "3d":
            self.no_signal_asimov_dataset = AsimovMapDataset.from_MapDataset(
                self.d_no_bkg
            )
        elif self.analysis == "1d":
            self.no_signal_asimov_dataset = AsimovSpectralDataset.from_SpectralDataset(
                self.d_no_bkg
            )
        self.no_signal_statistic = STATISTICS[self.stat_class_name](
            self.no_signal_asimov_dataset, poi_name=self.poi_name
        )
        # store the poi_ul somewhere
        self.poi_ul = None

    def compute(self):
        poi_best = self.statistic.poi_best

        if poi_best < 0:
            if self.mu_guess is not None:
                poi_ul = self.mu_guess
            else:
                poi_ul = 1e-2
        else:
            poi_ul = poi_best
        prev_pval = 0
        while (
            (pval := self.pvalue(poi_ul, cl_type=self.cl_type)) > 1 - self.cl
        ) or np.isnan(pval):
            prev_pval = pval
            poi_ul *= 2

        interp_ul_points = np.linspace(poi_ul / 2, poi_ul, 5)
        interp_pvalues = np.array(
            [
                self.pvalue(interp_ul, cl_type=self.cl_type)
                for interp_ul in interp_ul_points
            ]
        ).ravel()

        interpolation = interp1d(
            interp_ul_points, interp_pvalues - 1 + self.cl, kind="quadratic"
        )

        poi_ul = brentq(interpolation, poi_ul / 2, poi_ul)

        print("FOUND:", poi_ul)
        self.poi_ul = poi_ul

        return poi_ul

    def pvalue(self, poi_ul, cl_type):
        pval_bkg = 0
        if self.statistic.__class__.__name__ == "QTildeMuTestStatistic":
            ts_val = self.statistic.evaluate(poi_ul)  # ts_val on measurement_dataset

            if cl_type == "s":
                ts_val_bkg_asi = self.no_signal_statistic.evaluate(poi_ul)
                return (1 - norm.cdf(np.sqrt(ts_val))) / norm.cdf(
                    np.sqrt(ts_val_bkg_asi) - np.sqrt(ts_val)
                )
            else:
                return 1 - norm.cdf(np.sqrt(ts_val))

        else:
            pval_sig_bkg = self.statistic.pvalue(poi_ul)

            if cl_type == "s":
                pval_bkg = self.statistic.pvalue(0)

        return pval_sig_bkg / (1 - pval_bkg)

    def expected_uls(self):
        # scan for poi_ul median
        poi_ul = 1e-2
        target_ts = norm.ppf(1 - 0.5 * (1 - self.cl)) ** 2
        while (ts := self.no_signal_statistic.evaluate(poi_ul)) < target_ts:
            poi_ul *= 2

        interp_ul_points = np.linspace(poi_ul / 2, poi_ul, 10)
        interp_ts = np.array(
            [
                self.no_signal_statistic.evaluate(interp_ul)
                for interp_ul in interp_ul_points
            ]
        ).ravel()

        interpolation = interp1d(
            interp_ul_points, interp_ts - target_ts, kind="quadratic"
        )

        poi_ul = brentq(interpolation, poi_ul / 2, poi_ul)

        sigma = poi_ul / np.sqrt(target_ts)

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
            return sigma * (norm.ppf(1 - (1 - self.cl) * norm.cdf(n_sigma)) + n_sigma)


class ULFactory:
    def __init__(
        self,
        measurement_dataset,
        channels,
        mass_min,
        mass_max,
        n_steps,
        jfactor,
        analysis="3d",
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
        self.jfactor = jfactor
        self.analysis = analysis
        self.cl_type = cl_type
        self.cl = cl
        self.kwargs = kwargs
        self.max_workers = max_workers
        self.kwargs["cl"] = self.cl
        self.kwargs["cl_type"] = self.cl_type
        self.kwargs["analysis"] = self.analysis
        self.uls = None
        self.expected_uls = None

    def setup_models(self):
        models = []
        if self.analysis == "3d":
            spatial_model = TemplateSpatialModel(self.jfactor, normalize=False)

            if self.measurement_dataset.background_model:
                bkg_model = self.measurement_dataset.background_model.copy()
            else:
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

        elif self.analysis == "1d":
            if self.measurement_dataset.background_model:
                bkg_model = self.measurement_dataset.background_model.copy()
            else:
                bkg_model = FoVBackgroundModel(dataset_name="foo")

            for channel in self.channels:
                for mass in self.masses:
                    spectral_model = DarkMatterAnnihilationSpectralModel(
                        mass=mass * u.TeV, channel=channel, jfactor=self.jfactor
                    )
                    sky_model = SkyModel(
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
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(self.setup_calculator(models).compute)
                for models in self.setup_models()
            ]
            uls = [future.result() for future in futures]
        self.uls = uls

    def compute_expected(self):
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(self.setup_calculator(models).expected_uls)
                for models in self.setup_models()
            ]
            expected_uls = [future.result() for future in futures]
        self.expected_uls = expected_uls

    def compute(self):
        self.compute_uls()
        self.compute_expected()

    def write(self, path, overwrite=False, **kwargs):
        if self.uls is None and self.expected_uls is None:
            raise ValueError("No results computed yet. Run compute() first.")

        n_channels = len(self.channels)
        # prepare uls
        if self.uls is not None:
            uls = np.array(self.uls).reshape(n_channels, -1)

        # prepare expected uls
        if self.expected_uls is not None:
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
            ul_dict = {
                "mass": self.masses * u.TeV,
                "channel": np.repeat(channel, len(self.masses)),
                "cl_type": np.repeat(self.cl_type, len(self.masses)),
                "cl": np.repeat(self.cl, len(self.masses)),
            }

            if self.uls is not None:
                ul_dict["ul"] = uls[ch_idx] * CS

            if self.expected_uls is not None:
                ul_dict_expected = {
                    "median_ul": median_uls[ch_idx] * CS,
                    "1sigma_minus_ul": one_sigma_minus_uls[ch_idx] * CS,
                    "1sigma_plus_ul": one_sigma_plus_uls[ch_idx] * CS,
                    "2sigma_minus_ul": two_sigma_minus_uls[ch_idx] * CS,
                    "2sigma_plus_ul": two_sigma_plus_uls[ch_idx] * CS,
                }
                ul_dict = ul_dict | ul_dict_expected

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
