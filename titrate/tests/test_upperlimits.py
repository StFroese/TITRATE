import numpy as np
import pytest


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
