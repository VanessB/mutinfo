import numpy
import math

from collections import defaultdict, Counter
from collections.abc import Sequence
from scipy.stats import randint
from scipy.stats._distn_infrastructure import rv_frozen, rv_discrete_frozen
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
            Random non-repetitive sampling.
        """

        if label is None:
            return numpy.random.choice(len(self._dataset), size=size, replace=False)

        if not (label in self.subsets_indices.keys()):
            raise IndexError(f"The label `{label}` is not in the dataset.")

        indices = numpy.random.choice(len(self.subsets_indices[label]), size=size, replace=False)
        
        return self.subsets_indices[label][indices]


class mixed_by_label(multi_rv_frozen):
    def __init__(
        self,
        marginal_distributions: list[list[multi_rv_frozen | rv_frozen]],
        labels_distribution: rv_discrete_frozen
    ) -> None:
        self._marginal_distributions = marginal_distributions
        self._labels_distribution = labels_distribution

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

        labels_tuple = self._labels_distribution.rvs(size=size)
        
        # Cannot come up with a better way to do this.
        sampling = []
        for labels, marginal_distribution in zip(labels_tuple, self._marginal_distributions):
            labels_counts = Counter(labels)
            labels_invargsort = numpy.empty_like(labels)
            labels_invargsort[numpy.argsort(labels, kind="stable")] = numpy.arange(len(labels))

            sampling.append(
                numpy.concatenate(
                    [
                        marginal_distribution[label].rvs(size=labels_counts[label])
                        for label in sorted(labels_counts.keys())
                    ],
                    axis=0
                )[labels_invargsort]
            )

        return tuple(sampling)

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """

        if len(self._marginal_distributions) != 2:
            raise ValueError("Mutual information is only defined for pairs of random variables.")
        
        return self._labels_distribution.mutual_information()