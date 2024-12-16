import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import visualization as viz
from astropy.table import QTable, unique
from astropy.units import Quantity
from gammapy.modeling import Fit

from titrate.datasets import AsimovMapDataset, AsimovSpectralDataset
from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic
from titrate.utils import copy_dataset_with_models


STATISTICS = {"qmu": QMuTestStatistic, "qtildemu": QTildeMuTestStatistic}


class UpperLimitPlotter:
    def __init__(self, path, channel, show_uls=True, show_expected_uls=True, ax=None):
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

        if not show_uls and not show_expected_uls:
            raise ValueError("Either uls or expected_uls must be True")

        masses = table["mass"]
        if show_uls:
            try:
                uls = table["ul"]
            except KeyError:
                raise KeyError("No upper limits in dataframe. Set uls=False")
        else:
            uls = None

        if show_expected_uls:
            try:
                median = table["median_ul"]
                one_sigma_minus = table["1sigma_minus_ul"]
                one_sigma_plus = table["1sigma_plus_ul"]
                two_sigma_minus = table["2sigma_minus_ul"]
                two_sigma_plus = table["2sigma_plus_ul"]
            except KeyError:
                raise KeyError(
                    "No expected upper limits in dataframe. Set expected_uls=False"
                )
        else:
            median = None
            one_sigma_minus = None
            one_sigma_plus = None
            two_sigma_minus = None
            two_sigma_plus = None

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
        ul_unit = uls.unit if show_uls else median.unit
        self.ax.set_ylabel(
            rf"$CL_{cl_type}^{{{cl}}}$ upper limit on $< \sigma v>$ / {ul_unit:latex}"
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
        if uls is not None:
            self.ax.plot(masses, uls, color="C1", label="Upper Limits")
        if median is not None:
            self.ax.plot(masses, median, color="C0", label="Expected Upper Limits")
            self.ax.fill_between(
                masses,
                median,
                one_sigma_plus,
                color="C0",
                alpha=0.75,
                label=r"$1\sigma$-region",
            )
            self.ax.fill_between(
                masses, median, one_sigma_minus, color="C0", alpha=0.75
            )
            self.ax.fill_between(
                masses,
                one_sigma_plus,
                two_sigma_plus,
                color="C0",
                alpha=0.5,
                label=r"$2\sigma$-region",
            )
            self.ax.fill_between(
                masses, one_sigma_minus, two_sigma_minus, color="C0", alpha=0.5
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
        analysis="3d",
    ):
        self.path = path
        self.ax = ax if ax is not None else plt.gca()
        self.analysis = analysis
        self.measurement_dataset = measurement_dataset
        self.poi_name = poi_name

        self.d_sig = copy_dataset_with_models(self.measurement_dataset)
        self.d_sig.models.parameters[self.poi_name].value = 1e5
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

        try:
            table_diff = QTable.read(
                self.path, path=f"validation/{statistic}/diff/{channel}/{mass}"
            )
            table_same = QTable.read(
                self.path, path=f"validation/{statistic}/same/{channel}/{mass}"
            )
        except OSError:
            if channel is None:
                channels = list(
                    h5py.File(self.path)["validation"][statistic]["diff"].keys()
                )
                channels = [ch for ch in channels if "meta" not in ch]
                raise ValueError(f"Channel must be one of {channels}")
            if mass is None:
                masses = list(
                    h5py.File(self.path)["validation"][statistic]["diff"][
                        channel
                    ].keys()
                )
                masses = [Quantity(m) for m in masses if "meta" not in m]
                raise ValueError(f"Mass must be one of {masses}")

        toys_ts_diff = table_diff["ts"]
        toys_ts_diff_valid = table_diff["valid"]
        toys_ts_same = table_same["ts"]
        toys_ts_same_valid = table_same["valid"]

        # apply masks
        toys_ts_diff = toys_ts_diff[toys_ts_diff_valid]
        toys_ts_same = toys_ts_same[toys_ts_same_valid]

        max_ts = max(toys_ts_diff.max(), toys_ts_same.max())
        bins = np.linspace(0, max_ts, 31)
        linspace = np.linspace(0, max_ts, 1000)
        statistic_sig = STATISTICS[statistic](self.asimov_sig_dataset, poi_name)
        statistic_bkg = STATISTICS[statistic](self.asimov_bkg_dataset, poi_name)
        statistic_math_name = (
            r"q_\mu" if isinstance(statistic, QMuTestStatistic) else r"\tilde{q}_\mu"
        )

        self.plot(
            linspace,
            bins,
            toys_ts_same,
            toys_ts_diff,
            statistic_sig,
            statistic_bkg,
            statistic_math_name,
        )

        self.ax.set_yscale("log")
        self.ax.set_xlim(0, max_ts)

        self.ax.set_ylabel(r"$f(\tilde{q}_\mu\vert\mu^\prime,\theta_{\mu,\text{obs}})$")
        self.ax.set_xlabel(rf"${statistic_math_name}$")
        self.ax.set_title(statistic.__class__.__name__)
        self.ax.legend()

    def plot(
        self,
        linspace,
        bins,
        toys_ts_same,
        toys_ts_diff,
        statistic_sig,
        statistic_bkg,
        statistic_math_name,
    ):
        plt.hist(
            toys_ts_diff,
            bins=bins,
            density=True,
            histtype="step",
            color="C0",
            label=(r"$\mu=10^5$, $\mu^\prime=0$"),
        )
        plt.hist(
            toys_ts_same,
            bins=bins,
            density=True,
            histtype="step",
            color="C1",
            label=(r"$\mu=10^5$, $\mu^\prime=10^5$"),
        )

        plt.plot(
            linspace,
            statistic_bkg.asympotic_approximation_pdf(
                poi_val=1e5, same=False, poi_true_val=0, ts_val=linspace
            ),
            color="C0",
            label=r"$\mu=10^5$, $\mu^\prime=0$",
        )
        plt.plot(
            linspace,
            statistic_sig.asympotic_approximation_pdf(poi_val=1e5, ts_val=linspace),
            color="C1",
            ls="--",
            label=r"$\mu=10^5$, $\mu^\prime=10^5$",
        )
