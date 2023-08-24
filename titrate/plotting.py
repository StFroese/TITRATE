import matplotlib.pyplot as plt
from astropy import visualization as viz
from astropy.table import QTable, unique


class UpperLimitPlotter:
    def __init__(self, path, channel, axes=None):
        self.path = path
        self.axes = axes if axes is not None else plt.gca()

        table = QTable.read(self.path, path="upperlimits")

        if channel not in unique(table, keys="channel")["channel"]:
            raise KeyError(
                f"Channel {channel} not in dataframe."
                f'Choose from {unique(table, keys="channel")["channel"].tolist()}'
            )
        self.channel = channel

        masses = unique(table, keys="mass")["mass"]
        uls = table[table["channel"] == self.channel]["ul"]
        median = table[table["channel"] == self.channel]["median_ul"]
        one_sigma_minus = table[table["channel"] == self.channel]["1sigma_minus_ul"]
        one_sigma_plus = table[table["channel"] == self.channel]["1sigma_plus_ul"]
        two_sigma_minus = table[table["channel"] == self.channel]["2sigma_minus_ul"]
        two_sigma_plus = table[table["channel"] == self.channel]["2sigma_plus_ul"]

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

        self.axes.set_xscale("log")
        self.axes.set_yscale("log")

        cl_type = unique(table[table["channel"] == self.channel], keys="cl_type")[
            "cl_type"
        ][0]
        cl = unique(table[table["channel"] == self.channel], keys="cl")["cl"][0]
        self.axes.set_xlabel(f"m / {masses.unit:latex}")
        self.axes.set_ylabel(
            rf"$CL_{cl_type}^{{{cl}}}$ upper limit on $< \sigma v>$ / {uls.unit:latex}"
        )

        self.axes.set_title(f"Annihilation Upper Limits for channel {self.channel}")

        self.axes.legend()

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
        self.axes.plot(masses, uls, color="tab:orange", label="Upper Limits")
        self.axes.plot(masses, median, color="tab:blue", label="Expected Upper Limits")
        self.axes.fill_between(
            masses,
            median,
            one_sigma_plus,
            color="tab:blue",
            alpha=0.75,
            label=r"$1\sigma$-region",
        )
        self.axes.fill_between(
            masses, median, one_sigma_minus, color="tab:blue", alpha=0.75
        )
        self.axes.fill_between(
            masses,
            one_sigma_plus,
            two_sigma_plus,
            color="tab:blue",
            alpha=0.5,
            label=r"$2\sigma$-region",
        )
        self.axes.fill_between(
            masses, one_sigma_minus, two_sigma_minus, color="tab:blue", alpha=0.5
        )
