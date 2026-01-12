import numpy
import math

from collections.abc import Callable
from scipy.stats._distn_infrastructure import rv_frozen, rv_discrete_frozen
from scipy.stats._multivariate import multi_rv_frozen
from typing import Any


class modulated(multi_rv_frozen):
    def __init__(
        self,
        marginal_distributions: list[multi_rv_frozen | rv_frozen],
        modulation_distribution: multi_rv_frozen | rv_frozen,
        modulator: Callable[[list[Any], list[Any]], Any]
    ) -> None:
        self._marginal_distributions = marginal_distributions
        self._modulation_distribution = modulation_distribution
        self._modulator = modulator

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

        marginal_samples = [distribution.rvs(size) for distribution in self._marginal_distributions]
        modulation_samples = self._modulation_distribution.rvs(size)

        return self._modulator(marginal_samples, modulation_samples)

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
        
        return self._modulation_distribution.mutual_information