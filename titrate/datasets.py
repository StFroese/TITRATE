import numpy as np
from gammapy.datasets import MapDataset


class AsimovMapDataset(MapDataset):
    """
    AsimovMapDataset is a subclass of a gammapy MapDataset
    and provides asimov-like fake method.
    """

    def fake(self):
        """
        This method generates Asimov like counts,
        i.e. the counts are not drawn from a poisson distribution.
        """
        npred = self.npred()
        data = np.nan_to_num(npred.data, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        npred.data = data
        self.counts = npred
