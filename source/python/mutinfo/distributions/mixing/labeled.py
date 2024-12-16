import numpy
import math

from collections import defaultdict, Counter
from collections.abc import Sequence
from scipy.stats import randint
from scipy.stats._distn_infrastructure import rv_discrete_frozen
from scipy.stats._multivariate import multi_rv_frozen

class selector(multi_rv_frozen):
    """
    Frozen distribution with label data.
    """
    
    def __init__(self, dataset: Sequence) -> None:
        self._dist = randint
        self._dataset = dataset

        self.subsets_indices = defaultdict(list)
        for index, (value, label) in enumerate(self._dataset):
            self.subsets_indices[label].append(index)

        # Convert to numpy.ndarray.
        self.labels = list(self.subsets_indices.keys())
        self.subsets_indices = {key: numpy.array(value) for key, value in self.subsets_indices.items()}

    def rvs(self, size: int=1, label=None) -> list:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.
        label : otional
            A label to condition sampling.

        Returns
        -------
        x : numpy.ndarray
            Random sampling.
        """

        if label is None:
            indices = self._dist.rvs(size=size, low=0, high=len(self._dataset))
            #return [self.dataset[index][0] for index in indices]
            return indices

        if not (label in self.subsets_indices.keys()):
            raise IndexError(f"The label `{label}` is not present in the dataset.")

        indices = self._dist.rvs(size=size, low=0, high=len(self.subsets_indices[label]))
        #return [self.dataset[self.label_indices[label][index]][0] for index in indices]
        return self.subsets_indices[label][indices]


class paired_by_label(multi_rv_frozen):
    def __init__(
        self,
        datasets: list[Sequence],
        label_distribution: rv_discrete_frozen
    ) -> None:
        self._dist = label_distribution
        self._selectors = [selector(dataset) for dataset in datasets]

    def rvs(self, size: int=1) -> list:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x_1, ..., x_k : numpy.ndarray
            Random sampling.
        """

        label_indices = self._dist.rvs(size=size)
        label_indices_counts = Counter(label_indices)

        # OMG, this have to be redone.
        sampling = [[] for _ in self._selectors]
        for label_index in label_indices_counts.keys():
            for index, selector in enumerate(self._selectors):
                sampling[index].append(
                    selector.rvs(size=label_indices_counts[label_index], label=selector.labels[label_index])
                )

        return tuple(numpy.concatenate(_, axis=0) for _ in sampling)

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """

        # TODO: proper error.
        assert len(self._selectors) == 2
        
        return self._dist.entropy()