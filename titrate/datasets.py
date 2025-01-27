import numpy as np
from gammapy.datasets import MapDataset, SpectrumDataset

from titrate.utils import copy_models_to_dataset


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
        self.counts.data = npred.data.copy()

    @classmethod
    def from_MapDataset(self, dataset):
        dataset_dict = dataset.__dict__.copy()

        delete_keys = [key for key in dataset_dict.keys() if key.startswith("_")]

        deleted_entries = {}
        for key in delete_keys:
            deleted_entries[key] = dataset_dict.pop(key)

        asimov_dataset = AsimovMapDataset(**dataset_dict)
        for key in deleted_entries.keys():
            if key == "_name":
                continue
            setattr(asimov_dataset, key, deleted_entries[key])

        copy_models_to_dataset(dataset.models, asimov_dataset)
        asimov_dataset.fake()

        return asimov_dataset


class AsimovSpectralDataset(SpectrumDataset):
    """
    AsimovMapDataset is a subclass of a gammapy SpectrumDataset
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
        self.counts = npred.data.copy()

    @classmethod
    def from_SpectralDataset(self, dataset):
        dataset_dict = dataset.__dict__.copy()

        delete_keys = [key for key in dataset_dict.keys() if key.startswith("_")]

        deleted_entries = {}
        for key in delete_keys:
            deleted_entries[key] = dataset_dict.pop(key)

        asimov_dataset = AsimovSpectralDataset(**dataset_dict)

        copy_models_to_dataset(dataset.models, asimov_dataset)
        asimov_dataset.fake()

        return asimov_dataset
