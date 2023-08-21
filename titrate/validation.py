from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from titrate.statistics import QMuTestStatistic, QTildeMuTestStatistic, kstest
from titrate.utils import calc_ts_toyMC

STATISTICS = {"qmu": QMuTestStatistic, "qtildemu": QTildeMuTestStatistic}


class AsymptoticValidator:
    def __init__(
        self, measurement_dataset, asimov_dataset, statistic="qmu", poi_name=""
    ):
        if statistic not in STATISTICS.keys():
            raise ValueError(
                "Statistic must be one of {}".format(list(STATISTICS.keys()))
            )
        self.statistic_key = statistic
        self.statistic = STATISTICS[statistic]
        self.measurement_dataset = measurement_dataset
        self.asimov_dataset = asimov_dataset
        self.poi_name = poi_name

    def validate(self, n_toys=1000):
        toys_ts_diff = self.toys_ts(n_toys, 1, 0)
        toys_ts_same = self.toys_ts(n_toys, 1, 1)

        # only validate ts values above zero because
        # QTildeMuTestStatistic cdf will have problems with negative values in sqrt
        toys_ts_diff = toys_ts_diff[toys_ts_diff >= 0]
        toys_ts_same = toys_ts_same[toys_ts_same >= 0]

        stat = self.statistic(self.asimov_dataset, self.poi_name)
        ks_diff = kstest(
            toys_ts_diff,
            lambda x: stat.asympotic_approximation_cdf(
                poi_val=1, same=False, poi_true_val=0, ts_val=x
            ),
        )
        ks_same = kstest(
            toys_ts_same,
            lambda x: stat.asympotic_approximation_cdf(poi_val=1, ts_val=x),
        )

        valid = ks_diff > 0.05 and ks_same > 0.05

        return {"pvalue_diff": ks_diff, "pvalue_same": ks_same, "valid": valid}

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

    def plot_validation(self, n_toys=1000):
        toys_ts_diff = self.toys_ts(n_toys, 1, 0)
        toys_ts_same = self.toys_ts(n_toys, 1, 1)

        max_q = max(toys_ts_diff.max(), toys_ts_same.max())
        bins = np.linspace(0, max_q, 31)
        plt.hist(
            toys_ts_diff,
            bins=bins,
            density=True,
            histtype="step",
            color="tab:blue",
            label=r"$f(q_\mu\vert\mu^\prime)$, poi_val=1, poi_true_val=0",
        )
        plt.hist(
            toys_ts_same,
            bins=bins,
            density=True,
            histtype="step",
            color="tab:orange",
            label=r"$f(q_\mu\vert\mu)$, poi_val=1, poi_true_val=1",
        )

        lin_q = np.linspace(0, max_q, 1000)
        stat = self.statistic(self.asimov_dataset, self.poi_name)

        plt.plot(
            lin_q,
            stat.asympotic_approximation_pdf(
                poi_val=1, same=False, poi_true_val=0, ts_val=lin_q
            ),
            color="tab:blue",
            label=r"$f(q_\mu\vert\mu^\prime)$, asympotic",
        )
        plt.plot(
            lin_q,
            stat.asympotic_approximation_pdf(poi_val=1, ts_val=lin_q),
            color="tab:orange",
            label=r"$f(q_\mu\vert\mu)$, asympotic",
        )

        plt.yscale("log")
        plt.xlim(0, max_q)

        plt.ylabel("pdf")
        plt.xlabel("q")
        plt.title(self.statistic.__name__)
        plt.legend()
        plt.show()
