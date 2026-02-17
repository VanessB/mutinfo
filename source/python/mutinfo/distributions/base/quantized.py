import math
import numpy

from scipy.stats import uniform, randint
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from ..tools import BaseMutualInformationTest
from ...utils.checks import _check_dimension_value, _check_mutual_information_value, _check_probability_value


class quantized(multi_rv_frozen, BaseMutualInformationTest):
    """
    Frozen quantized distribution with known mutual information.
    """
    
    def __init__(
        self,
        base_rv: rv_frozen,
        quantiles: numpy.ndarray,
        normalize: bool=False,
        *args, **kwargs
    ) -> None:
        """
        Create a frozen quantized distribution with known mutual information.

        Parameters
        ----------
        base_rv : scipy.stats._multivariate.multi_rv_frozen
            Base distribution.
        quantiles : array_like
            Quantiles to be used to assign labels.
        normalize : bool, optional
            Divide the labels by the total number of possible outcomes.
            Default: False
        """
        
        multi_rv_frozen.__init__(self, *args, **kwargs)

        try:
            quantiles = numpy.asarray(quantiles)
        except:
            TypeError("Expected `quantiles` to be convertible to `numpy.ndarray`")

        if len(quantiles) != 1:
            TypeError("Expected `quantiles` to be an 1D array")
            
        _check_probability_value(quantiles, "quantiles")

        self._quantiles = numpy.sort(quantiles)
        self._dist = base_rv

        self.normalize = normalize

    def rvs(self, *args, **kwargs) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Random variate.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.

        Returns
        -------
        x, y : numpy.ndarray
            Random sampling.
        """
        
        x = self._dist.rvs(*args, **kwargs)
        y = numpy.sum(self._dist.cdf(x)[:,numpy.newaxis,...] > self._quantiles.reshape((1,-1) + (1,) * (len(x.shape)-1)), axis=1)

        if self.normalize:
            y = y / self._quantiles.shape[0]
        
        return x, y

    @property
    def label_probabilities(self) -> numpy.ndarray:
        """
        Probabilities of the labels.

        Returns
        -------
        labels_probabilities : numpy.ndarray
            Array of probabilities of the labels.
        """
        
        padded_quantiles = numpy.pad(self._quantiles, pad_width=(1, 1), constant_values=(0.0, 1.0))
        
        return padded_quantiles[1:] - padded_quantiles[:-1]

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        
        label_probabilities = self.label_probabilities
        
        return -numpy.sum(xlogy(label_probabilities, label_probabilities))


class smoothed_quantized(multi_rv_frozen, BaseMutualInformationTest):
    def __init__(
        self,
        alphabet_size: int,
        smoothing_epsilon: int,
        normalize: bool=False,
        *args, **kwargs
    ) -> None:
        """
        Create a correlated variables from the mixture of dicrete and continuous
        distributions with known mutual information.

        Parameters
        ----------
        alphabet_size : int >= 2
            Discrete variable support size.
        smoothing_epsilon : int >= 1
            Additive noise support length.
        normalize : bool, optional
            Divide the labels by the total number of possible outcomes.
            Default: False
        """

        multi_rv_frozen.__init__(self, *args, **kwargs)

        if not isinstance(alphabet_size, int):
            raise TypeError("Expected `alphabet_size` of type `int`")
        if alphabet_size <= 1:
            raise ValueError("Expected `alphabet_size` to be greater than 1")

        if not isinstance(smoothing_epsilon, int):
            raise TypeError("Expected `smoothing_epsilon` of type `int`")
        if smoothing_epsilon <= 0:
            raise ValueError("Expected `smoothing_epsilon` to be greater than 0")

        if alphabet_size < smoothing_epsilon - 1:
            raise ValueError("Expected `alphabet_size` to be greater than `smoothing_epsilon` - 2")

        self.alphabet_size = alphabet_size
        self.smoothing_epsilon = smoothing_epsilon
        self.normalize = normalize

        self.x_dist = randint(0, self.alphabet_size)
        self.z_dist = uniform(0, self.smoothing_epsilon)

    def rvs(self, size: int=1, *args, **kwargs) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x, y : tuple[numpy.ndarray, numpy.ndarray]
            Random sampling.
        """

        x = self.x_dist.rvs(size)
        y = x + self.z_dist.rvs(size)

        if self.normalize:
            x = x / (self.alphabet_size - 1)
            y = y / (self.alphabet_size - 1 + self.smoothing_epsilon)

        return x, y

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        return (
            self.smoothing_epsilon * (self.smoothing_epsilon - 1) * math.log(self.alphabet_size * self.smoothing_epsilon) -
            2.0 * sum(i * math.log(i) for i in range(1, self.smoothing_epsilon))
        ) / (self.alphabet_size * self.smoothing_epsilon) + (
            (self.alphabet_size - self.smoothing_epsilon + 1) * math.log(self.alphabet_size) / self.alphabet_size
        ) - math.log(self.smoothing_epsilon)