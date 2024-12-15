from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import h5py
import numpy as np
from astropy.table import QTable
from astropy.units import Quantity
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel

from titrate.datasets import AsimovMapDataset, AsimovSpectralDataset
from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic, kstest
from titrate.utils import calc_ts_toyMC, copy_dataset_with_models

STATISTICS = {"qmu": QMuTestStatistic, "qtildemu": QTildeMuTestStatistic}


class AsymptoticValidator:
    def __init__(
        self,
        measurement_dataset,
        statistic="qmu",
        poi_name="scale",
        path=None,
        channel=None,
        mass=None,
        max_workers=None,
        analysis="3d",
    ):
        if statistic not in STATISTICS.keys():
            raise ValueError(
                "Statistic must be one of {}".format(list(STATISTICS.keys()))
            )
        self.statistic_key = statistic
        self.statistic = STATISTICS[statistic]
        self.poi_name = poi_name

        self.measurement_dataset = measurement_dataset
        self.d_sig = copy_dataset_with_models(self.measurement_dataset)
        self.d_sig.models.parameters[self.poi_name].value = 1
        self.d_sig.models.parameters[self.poi_name].frozen = True
        fit = Fit()
        _ = fit.run(self.d_sig)
        self.d_sig.models.parameters[self.poi_name].frozen = False

        self.d_bkg = copy_dataset_with_models(self.measurement_dataset)
        self.d_bkg.models.parameters[self.poi_name].value = 0
        self.d_bkg.models.parameters[self.poi_name].frozen = True
        fit = Fit()
        _ = fit.run(self.d_bkg)
        self.d_bkg.models.parameters[self.poi_name].frozen = False

        self.analysis = analysis
        if self.analysis == "3d":
            self.asimov_sig_dataset = AsimovMapDataset.from_MapDataset(self.d_sig)
            self.asimov_bkg_dataset = AsimovMapDataset.from_MapDataset(self.d_bkg)
        elif self.analysis == "1d":
            self.asimov_sig_dataset = AsimovSpectralDataset.from_SpectralDataset(
                self.d_sig
            )
            self.asimov_bkg_dataset = AsimovSpectralDataset.from_SpectralDataset(
                self.d_bkg
            )

        self.path = path
        self.channel = channel
        self.mass = mass
        if self.channel is None and self.path is not None:
            channels = list(
                h5py.File(self.path)["validation"][self.statistic_key]["diff"].keys()
            )
            channels = [ch for ch in channels if "meta" not in ch]
            raise ValueError(f"Channel must be one of {channels}")
        if self.mass is None and self.path is not None:
            masses = list(
                h5py.File(self.path)["validation"][self.statistic_key]["diff"][
                    self.channel
                ].keys()
            )
            masses = [Quantity(m) for m in masses if "meta" not in m]
            raise ValueError(f"Mass must be one of {masses}")

        self.max_workers = max_workers

        self.toys_ts_diff = None
        self.toys_ts_same = None
        self.toys_ts_diff_valid = None
        self.toys_ts_same_valid = None

    def validate(self, n_toys=1000):
        self.generate_datasets(n_toys)

        stat_sig = self.statistic(self.asimov_sig_dataset, self.poi_name)
        stat_bkg = self.statistic(self.asimov_bkg_dataset, self.poi_name)
        ks_diff = kstest(
            self.toys_ts_diff[self.toys_ts_diff_valid],
            lambda x: stat_bkg.asympotic_approximation_cdf(
                poi_val=1, same=False, poi_true_val=0, ts_val=x
            ),
        )
        ks_same = kstest(
            self.toys_ts_same[self.toys_ts_same_valid],
            lambda x: stat_sig.asympotic_approximation_cdf(poi_val=1, ts_val=x),
        )

        valid = ks_diff > 0.05 and ks_same > 0.05

        return {"pvalue_diff": ks_diff, "pvalue_same": ks_same, "valid": valid}

    def generate_datasets(self, n_toys):
        if self.path is None:
            toys_ts_diff, toys_ts_diff_valid = self.toys_ts(n_toys, 1, 0)
            toys_ts_same, toys_ts_same_valid = self.toys_ts(n_toys, 1, 1)
        else:
            (
                toys_ts_diff,
                toys_ts_diff_valid,
                toys_ts_same,
                toys_ts_same_valid,
            ) = self.open_toys()

        self.toys_ts_diff = toys_ts_diff
        self.toys_ts_same = toys_ts_same
        self.toys_ts_diff_valid = toys_ts_diff_valid
        self.toys_ts_same_valid = toys_ts_same_valid

    @lru_cache
    def toys_ts(self, n_toys, poi_val, poi_true_val):
        with ProcessPoolExecutor(self.max_workers) as pool:
            futures = [
                pool.submit(
                    calc_ts_toyMC,
                    self.measurement_dataset,
                    self.statistic,
                    poi_val,
                    poi_true_val,
                    self.poi_name,
                )
                for _ in range(n_toys)
            ]
            toys_ts = [future.result() for future in futures]
            toys_valid = [True for _ in range(len(toys_ts))]

        # to ndarray
        toys_ts = np.array(toys_ts).ravel()
        toys_valid = np.array(toys_valid).ravel()

        return toys_ts, toys_valid

    def open_toys(self):
        toys_diff = QTable.read(
            self.path,
            path=f"validation/{self.statistic_key}/diff/{self.channel}/{self.mass}",
        )
        toys_same = QTable.read(
            self.path,
            path=f"validation/{self.statistic_key}/same/{self.channel}/{self.mass}",
        )

        toys_ts_diff = toys_diff["ts"]
        toys_ts_diff_valid = toys_diff["valid"]
        toys_ts_same = toys_same["ts"]
        toys_ts_same_valid = toys_same["valid"]

        return toys_ts_diff, toys_ts_diff_valid, toys_ts_same, toys_ts_same_valid

    def write(self, path, overwrite=False, **kwargs):
        if self.toys_ts_diff is None or self.toys_ts_same is None:
            raise ValueError("Toys not generated yet. Run validate() first.")

        # collect meta data
        for model in self.measurement_dataset.models:
            if isinstance(model, SkyModel):
                if isinstance(
                    model.spectral_model, DarkMatterAnnihilationSpectralModel
                ):
                    channel = model.spectral_model.channel
                    mass = model.spectral_model.mass
        try:
            channel
            mass
        except NameError:
            raise NameError(
                "Could not find channel and mass in measurement dataset. "
                "Please add a DarkMatterAnnihilationSpectralModel to the dataset."
            )

        # save toys
        toys_dict_diff = {
            "ts": self.toys_ts_diff,
            "valid": self.toys_ts_diff_valid,
        }
        toys_dict_same = {
            "ts": self.toys_ts_same,
            "valid": self.toys_ts_same_valid,
        }

        qtable = QTable(toys_dict_diff)
        qtable.write(
            path,
            format="hdf5",
            path=f"validation/{self.statistic_key}/diff/{channel}/{mass}",
            overwrite=overwrite,
            append=True,
            serialize_meta=True,
            **kwargs,
        )

        qtable = QTable(toys_dict_same)
        qtable.write(
            path,
            format="hdf5",
            path=f"validation/{self.statistic_key}/same/{channel}/{mass}",
            overwrite=overwrite,
            append=True,
            serialize_meta=True,
            **kwargs,
        )
