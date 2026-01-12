import numpy
import math

from collections import defaultdict, Counter
from collections.abc import Sequence
from scipy.stats import randint
from scipy.stats._distn_infrastructure import rv_frozen, rv_discrete_frozen
from scipy.stats._multivariate import multi_rv_frozen
from typing import Any


class subsampler(multi_rv_frozen):
    """
    Frozen distribution with label data.
    """
    
    def __init__(
        self,
        data: Sequence,
        subset_indices: numpy.ndarray,
        replace: bool=False
    ) -> None:
        self.data = data
        self.subset_indices = subset_indices
        self.replace = replace
        

    def rvs(self, size: int=1) -> Sequence:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x : numpy.ndarray
            Random non-repetitive sampling.
        """
            
        length = len(self.subset_indices)
        if not self.replace:
            length = (int(size / length) + 1) * length
        
        indices = numpy.random.choice(length, size=size, replace=self.replace)
        indices = numpy.remainder(indices, len(self.subset_indices)) # TODO: come up with something more effective.
        
        return self.data[self.subset_indices[indices]]


def labeled_dataset_to_subsamplers(
    data: numpy.ndarray,
    labels: numpy.ndarray,
    split_by_labels: bool=True,
) -> dict[Any, subsampler]:
    """
    Convert labeled data into a dict of per-class subsamplers.

    Parameters
    ----------
    data : array_like
        Labeled data.
    labels : array_like

    Returns
    -------
    x : numpy.ndarray
        Random non-repetitive sampling.
    """

    if split_by_labels:
        # Shitty as hell implementation.
        subsamplers = {}
        for label in numpy.unique(labels):
            subsamplers[label] = subsampler(data, numpy.nonzero(labels == label)[0])
    
        return subsamplers
    else:
        return subsampler(data, numpy.arange(0, len(data)))

# TODO: does it belong here?
def torchvision_default_transform(x: numpy.ndarray, to_CHW: bool=False) -> numpy.ndarray:   
    x = x / 255
    if len(x.shape) < 4:
        x = x[:,None,...]
    
    if to_CHW:
        x = x.transpose((0, 3, 1, 2))

    return x

# TODO: does it belong here?
def torchvision_labeled_dataset_to_subsamplers(
    dataset,
    transform=torchvision_default_transform,
    split_by_labels: bool=True,
) -> dict[Any, subsampler]:
    """
    Convert torchvision labeled dataset into a dict of per-class subsamplers.

    Parameters
    ----------
    data : array_like
        Labeled data.
    labels : array_like

    Returns
    -------
    x : numpy.ndarray
        Random non-repetitive sampling.
    """
    
    return labeled_dataset_to_subsamplers(
        transform(numpy.asarray(dataset.data)),
        numpy.asarray(dataset.targets),
        split_by_labels
    )


class mixed_by_label(multi_rv_frozen):
    def __init__(
        self,
        marginal_distributions: dict[Any, list[multi_rv_frozen | rv_frozen]],
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
        
        return self._labels_distribution.mutual_information