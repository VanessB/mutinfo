import math
import numpy

from scipy.stats import uniform
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from scipy.special import xlogy

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