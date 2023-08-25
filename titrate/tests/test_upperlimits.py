import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import QTable


@pytest.mark.parametrize("cl_type", ["s", "s+b"])
@pytest.mark.parametrize("statistic", ["qmu", "qtildemu"])
def test_ULCalculator(measurement_dataset, statistic, cl_type):
    from titrate.upperlimits import ULCalculator

    ulcalc = ULCalculator(
        measurement_dataset, statistic=statistic, poi_name="scale", cl_type=cl_type
    )
    ul = ulcalc.compute()
    bands = ulcalc.expected_uls()

    assert not np.isin(ul, [np.inf, np.nan])
    assert not np.isin(bands["med"], [np.inf, np.nan])
    assert not np.any(np.isin(bands["1sig"], [np.inf, np.nan]))
    assert not np.any(np.isin(bands["2sig"], [np.inf, np.nan]))

    assert bands["2sig"][0] < bands["1sig"][0]
    assert bands["med"] < bands["1sig"][1]
    assert bands["1sig"][1] < bands["2sig"][1]


@pytest.fixture(scope="module")
def upperlimits_file(jfact_map, measurement_dataset, tmp_path_factory):
    from titrate.upperlimits import ULFactory

    ulfactory = ULFactory(
        measurement_dataset, ["b", "W"], 0.1 * u.TeV, 100 * u.TeV, 5, jfact_map
    )
    ulfactory.compute()

    data = tmp_path_factory.mktemp("data")
    ulfactory.save_results(f"{data}/ul.hdf5")

    return f"{data}/ul.hdf5"


@pytest.mark.parametrize("channel", ["b", "W"])
def test_ULFactory(upperlimits_file, channel):
    table = QTable.read(upperlimits_file, path=channel)
    assert np.all(table["mass"] == np.geomspace(0.1, 100, 5) * u.TeV)
    assert len(table["ul"]) == 5
    assert len(table["median_ul"]) == 5
    assert len(table["1sigma_minus_ul"]) == 5
    assert len(table["1sigma_plus_ul"]) == 5
    assert len(table["2sigma_minus_ul"]) == 5
    assert len(table["2sigma_plus_ul"]) == 5
    assert len(table["cl_type"]) == 5
    assert np.all(table["cl_type"] == "s")
    assert len(table["cl"]) == 5
    assert np.all(table["cl"] == 0.95)
    assert np.all(table["channel"] == sorted([channel] * 5)[::-1])


def test_UpperLimitPlotter(upperlimits_file):
    from titrate.plotting import UpperLimitPlotter

    fig, axs = plt.subplots(nrows=1, ncols=2)

    for channel, ax in zip(["b", "W"], np.array(axs).reshape(-1)):
        UpperLimitPlotter(upperlimits_file, channel=channel, axes=ax)
