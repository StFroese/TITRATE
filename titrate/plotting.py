import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import visualization as viz
from astropy.table import QTable, unique
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
        poi_val=1e5,
    ):
        self.path = path
        self.ax = ax if ax is not None else plt.gca()
        self.analysis = analysis
        self.measurement_dataset = measurement_dataset
        self.poi_name = poi_name
        self.poi_val = poi_val

        self.d_sig = copy_dataset_with_models(self.measurement_dataset)
        self.d_sig.models.parameters[self.poi_name].value = self.poi_val
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

        table_diff = QTable.read(
            self.path, path=f"validation/{statistic}/diff/{channel}/{mass}"
        )
        table_same = QTable.read(
            self.path, path=f"validation/{statistic}/same/{channel}/{mass}"
        )

        toys_ts_diff = table_diff["ts"]
        toys_ts_diff_valid = table_diff["valid"]
        toys_ts_same = table_same["ts"]
        toys_ts_same_valid = table_same["valid"]

        # apply masks
        toys_ts_diff = toys_ts_diff[toys_ts_diff_valid]
        toys_ts_same = toys_ts_same[toys_ts_same_valid]

        bins_same = np.linspace(0, toys_ts_same.max(), 31)
        bins_diff = np.linspace(0, toys_ts_diff.max(), 31)
        linspace_same = np.linspace(1e-3, bins_same[-1], 1000)
        linspace_diff = np.linspace(1e-3, bins_diff[-1], 1000)
        statistic_sig = STATISTICS[statistic](self.asimov_sig_dataset, poi_name)
        statistic_bkg = STATISTICS[statistic](self.asimov_bkg_dataset, poi_name)
        statistic_math_name = (
            r"q_\mu" if isinstance(statistic, QMuTestStatistic) else r"\tilde{q}_\mu"
        )

        fig, axs = plt.subplot_mosaic([["same"], ["diff"]])

        self.plot(
            linspace_same,
            linspace_diff,
            bins_same,
            bins_diff,
            toys_ts_same,
            toys_ts_diff,
            statistic_sig,
            statistic_bkg,
            axs,
        )

        for ax in axs:
            axs[ax].set_yscale("log")
            axs[ax].set_xlim(
                0,
            )
            axs[ax].set_ylim(1e-4, 1e2)

            if ax == "same":
                axs[ax].set_ylabel(
                    r"$f(\tilde{q}_\mu\vert\mu^\prime,\theta_{\mu,\text{obs}})$ \\"
                    r"$\mu=10^5$, $\mu^\prime=10^5$"
                )
            else:
                axs[ax].set_ylabel(
                    r"$f(\tilde{q}_\mu\vert\mu^\prime,\theta_{\mu,\text{obs}})$ \\"
                    r"$\mu=10^5$, $\mu^\prime=0$"
                )
            axs[ax].set_xlabel(rf"${statistic_math_name}$")
            axs[ax].set_title(statistic.__class__.__name__)
            axs[ax].legend()
        self.fig = fig
        self.axs = axs

    def plot(
        self,
        linspace_same,
        linspace_diff,
        bins_same,
        bins_diff,
        toys_ts_same,
        toys_ts_diff,
        statistic_sig,
        statistic_bkg,
        axs,
    ):
        print(self.poi_val)
        axs["diff"].hist(
            toys_ts_diff,
            bins=bins_diff,
            density=True,
            histtype="step",
            color="C0",
            label=(r"MC"),
        )
        axs["same"].hist(
            toys_ts_same,
            bins=bins_same,
            density=True,
            histtype="step",
            color="C1",
            label=(r"MC"),
        )

        axs["diff"].plot(
            linspace_diff,
            statistic_bkg.asympotic_approximation_pdf(
                poi_val=self.poi_val, same=False, poi_true_val=0, ts_val=linspace_diff
            ),
            color="C0",
            ls="--",
            label=r"Asymptotic",
        )
        axs["same"].plot(
            linspace_same,
            statistic_sig.asympotic_approximation_pdf(
                poi_val=self.poi_val, ts_val=linspace_same
            ),
            color="C1",
            ls="--",
            label=r"Asymptotic",
        )
