import math
import numpy
from scipy.stats import norm

from scipy.stats._multivariate import multi_rv_frozen


def basic_conditioning(data: numpy.ndarray, condition_number: float) -> numpy.ndarray:
    if condition_number < 1.0:
        raise ValueError(f"Expected condition number to be not less than 1.0, got {condition_number}.")
    
    data = data.copy().reshape(data.shape[0], -1)
    dimensionality = data.shape[-1]

    data -= (1.0 - 1.0 / condition_number) * data.sum(axis=-1, keepdims=True) / dimensionality
    data *= math.sqrt(dimensionality / (dimensionality - (1 - 1.0 / condition_number**2)))

    return data


def total_conditioning(data: numpy.ndarray, condition_number: float) -> numpy.ndarray:
    if condition_number < 1.0:
        raise ValueError(f"Expected condition number to be not less than 1.0, got {condition_number}.")
    
    data = data.copy().reshape(data.shape[0], -1)
    dimensionality = data.shape[-1]

    data += (condition_number - 1.0) * data.sum(axis=-1, keepdims=True) / dimensionality
    data *= math.sqrt(dimensionality / (dimensionality + (condition_number**2 - 1.0)))

    return data


class AWGN(multi_rv_frozen):
    def __init__(
        self,
        base_rv,
        dimensionality: int,
        sigma: float,
        conditioning = None
    ) -> None:
        self._dist = base_rv
        self.dimensionality = dimensionality
        self.sigma = sigma
        self.conditioning = conditioning
        self._norm = norm(scale=sigma)

    def rvs(self, size: int=1) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x, y : numpy.ndarray
            Random sampling.
        """

        x = self._dist.rvs((size, self.dimensionality))

        if not (self.conditioning is None):
            x = self.conditioning(x)

        z = self._norm.rvs((size, self.dimensionality))

        return x, x + z