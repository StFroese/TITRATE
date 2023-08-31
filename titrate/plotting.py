import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import visualization as viz
from astropy.table import QTable, unique
from astropy.units import Quantity

from titrate.datasets import AsimovMapDataset
from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic

STATISTICS = {"qmu": QMuTestStatistic, "qtildemu": QTildeMuTestStatistic}


class UpperLimitPlotter:
    def __init__(self, path, channel, ax=None):
        self.path = path
        self.ax = ax if ax is not None else plt.gca()

        try:
            table = QTable.read(self.path, path=f"upperlimits/{channel}")
        except OSError:
            channels = list(h5py.File(self.path).keys())
            channels = [ch for ch in channels if "meta" not in ch]
            raise KeyError(
                f"Channel {channel} not in dataframe. " f"Choose from {channels}"
            )

        self.channel = channel

        masses = table["mass"]
        uls = table["ul"]
        median = table["median_ul"]
        one_sigma_minus = table["1sigma_minus_ul"]
        one_sigma_plus = table["1sigma_plus_ul"]
        two_sigma_minus = table["2sigma_minus_ul"]
        two_sigma_plus = table["2sigma_plus_ul"]

        with viz.quantity_support():
            self.plot_channel(
                masses,
                uls,
                median,
                one_sigma_minus,
                one_sigma_plus,
                two_sigma_minus,
                two_sigma_plus,
            )

        self.ax.set_xscale("log")
        self.ax.set_yscale("log")

        cl_type = unique(table[table["channel"] == self.channel], keys="cl_type")[
            "cl_type"
        ][0]
        cl = unique(table[table["channel"] == self.channel], keys="cl")["cl"][0]
        self.ax.set_xlabel(f"m / {masses.unit:latex}")
        self.ax.set_ylabel(
            rf"$CL_{cl_type}^{{{cl}}}$ upper limit on $< \sigma v>$ / {uls.unit:latex}"
        )

        self.ax.set_title(f"Annihilation Upper Limits for channel {self.channel}")

        self.ax.legend()

    def plot_channel(
        self,
        masses,
        uls,
        median,
        one_sigma_minus,
        one_sigma_plus,
        two_sigma_minus,
        two_sigma_plus,
    ):
        self.ax.plot(masses, uls, color="tab:orange", label="Upper Limits")
        self.ax.plot(masses, median, color="tab:blue", label="Expected Upper Limits")
        self.ax.fill_between(
            masses,
            median,
            one_sigma_plus,
            color="tab:blue",
            alpha=0.75,
            label=r"$1\sigma$-region",
        )
        self.ax.fill_between(
            masses, median, one_sigma_minus, color="tab:blue", alpha=0.75
        )
        self.ax.fill_between(
            masses,
            one_sigma_plus,
            two_sigma_plus,
            color="tab:blue",
            alpha=0.5,
            label=r"$2\sigma$-region",
        )
        self.ax.fill_between(
            masses, one_sigma_minus, two_sigma_minus, color="tab:blue", alpha=0.5
        )


class ValidationPlotter:
    def __init__(
        self,
        measurement_dataset,
        path,
        channel=None,
        mass=None,
        statistic="qmu",
        poi_name="scale",
        ax=None,
    ):
        self.path = path
        self.ax = ax if ax is not None else plt.gca()

        asimov_dataset = AsimovMapDataset.from_MapDataset(measurement_dataset)

        try:
            table = QTable.read(
                self.path, path=f"validation/{statistic}/{channel}/{mass}"
            )
        except OSError:
            if channel is None:
                channels = list(h5py.File(self.path)["validation"][statistic].keys())
                channels = [ch for ch in channels if "meta" not in ch]
                raise ValueError(f"Channel must be one of {channels}")
            if mass is None:
                masses = list(
                    h5py.File(self.path)["validation"][statistic][channel].keys()
                )
                masses = [Quantity(m) for m in masses if "meta" not in m]
                raise ValueError(f"Mass must be one of {masses}")

        toys_ts_same = table["toys_ts_same"]
        toys_ts_diff = table["toys_ts_diff"]

        max_ts = max(toys_ts_diff.max(), toys_ts_same.max())
        bins = np.linspace(0, max_ts, 31)
        linspace = np.linspace(0, max_ts, 1000)
        statistic = STATISTICS[statistic](asimov_dataset, poi_name)
        statistic_math_name = (
            r"q_\mu" if isinstance(statistic, QMuTestStatistic) else r"\tilde{q}_\mu"
        )

        self.plot(
            linspace, bins, toys_ts_same, toys_ts_diff, statistic, statistic_math_name
        )

        self.ax.set_yscale("log")
        self.ax.set_xlim(0, max_ts)

        self.ax.set_ylabel("pdf")
        self.ax.set_xlabel(rf"${statistic_math_name}$")
        self.ax.set_title(statistic.__class__.__name__)
        self.ax.legend()

    def plot(
        self, linspace, bins, toys_ts_same, toys_ts_diff, statistic, statistic_math_name
    ):
        plt.hist(
            toys_ts_diff,
            bins=bins,
            density=True,
            histtype="step",
            color="tab:blue",
            label=(
                rf"$f({statistic_math_name}\vert\mu^\prime)$, "
                r"$\mu=1$, $\mu^\prime=0$"
            ),
        )
        plt.hist(
            toys_ts_same,
            bins=bins,
            density=True,
            histtype="step",
            color="tab:orange",
            label=(
                rf"$f({statistic_math_name}\vert\mu^\prime)$, "
                r"$\mu=1$, $\mu^\prime=1$"
            ),
        )

        plt.plot(
            linspace,
            statistic.asympotic_approximation_pdf(
                poi_val=1, same=False, poi_true_val=0, ts_val=linspace
            ),
            color="tab:blue",
            label=rf"$f({statistic_math_name}\vert\mu^\prime)$, asympotic",
        )
        plt.plot(
            linspace,
            statistic.asympotic_approximation_pdf(poi_val=1, ts_val=linspace),
            color="tab:orange",
            label=rf"$f({statistic_math_name}\vert\mu^\prime)$, asympotic",
        )
