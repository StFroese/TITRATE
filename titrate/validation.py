from functools import lru_cache

import h5py
import numpy as np
from astropy.table import QTable
from astropy.units import Quantity
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel
from gammapy.modeling.models import SkyModel
from joblib import Parallel, delayed

from titrate.datasets import AsimovMapDataset
from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic, kstest
from titrate.utils import calc_ts_toyMC

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
    ):
        if statistic not in STATISTICS.keys():
            raise ValueError(
                "Statistic must be one of {}".format(list(STATISTICS.keys()))
            )
        self.statistic_key = statistic
        self.statistic = STATISTICS[statistic]

        self.measurement_dataset = measurement_dataset
        self.asimov_dataset = AsimovMapDataset.from_MapDataset(self.measurement_dataset)

        self.path = path
        self.channel = channel
        self.mass = mass
        if self.channel is None and self.path is not None:
            channels = list(
                h5py.File(self.path)["validation"][self.statistic_key].keys()
            )
            channels = [ch for ch in channels if "meta" not in ch]
            raise ValueError(f"Channel must be one of {channels}")
        if self.mass is None and self.path is not None:
            masses = list(
                h5py.File(self.path)["validation"][self.statistic_key][
                    self.channel
                ].keys()
            )
            masses = [Quantity(m) for m in masses if "meta" not in m]
            raise ValueError(f"Mass must be one of {masses}")

        self.poi_name = poi_name

        self.toys_ts_diff = None
        self.toys_ts_same = None

    def validate(self, n_toys=1000):
        self.generate_datasets(n_toys)

        stat = self.statistic(self.asimov_dataset, self.poi_name)
        ks_diff = kstest(
            self.toys_ts_diff,
            lambda x: stat.asympotic_approximation_cdf(
                poi_val=1, same=False, poi_true_val=0, ts_val=x
            ),
        )
        ks_same = kstest(
            self.toys_ts_same,
            lambda x: stat.asympotic_approximation_cdf(poi_val=1, ts_val=x),
        )

        valid = ks_diff > 0.05 and ks_same > 0.05

        return {"pvalue_diff": ks_diff, "pvalue_same": ks_same, "valid": valid}

    def generate_datasets(self, n_toys):
        if self.path is None:
            toys_ts_diff = self.toys_ts(n_toys, 1, 0)
            toys_ts_same = self.toys_ts(n_toys, 1, 1)
        else:
            toys_ts_same, toys_ts_diff = self.open_toys()

        self.toys_ts_diff = toys_ts_diff
        self.toys_ts_same = toys_ts_same

    @lru_cache
    def toys_ts(self, n_toys, poi_val, poi_true_val):
        toys_ts = Parallel(n_jobs=-1, verbose=0)(
            delayed(calc_ts_toyMC)(
                self.measurement_dataset,
                self.statistic,
                poi_val,
                poi_true_val,
                self.poi_name,
            )
            for _ in range(n_toys)
        )

        # to ndarray
        toys_ts = np.array(toys_ts).ravel()

        return toys_ts

    def open_toys(self):
        toys = QTable.read(
            self.path,
            path=f"validation/{self.statistic_key}/{self.channel}/{self.mass}",
        )

        toys_ts_diff = toys["toys_ts_diff"]
        toys_ts_same = toys["toys_ts_same"]

        return toys_ts_same, toys_ts_diff

    def save_toys(self, path, overwrite=False, **kwargs):
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
        toys_dict = {
            "toys_ts_diff": self.toys_ts_diff,
            "toys_ts_same": self.toys_ts_same,
        }

        qtable = QTable(toys_dict)
        qtable.write(
            path,
            format="hdf5",
            path=f"validation/{self.statistic_key}/{channel}/{mass}",
            overwrite=overwrite,
            append=True,
            serialize_meta=True,
            **kwargs,
        )
