import math
import numpy

from scipy.optimize import bisect
from scipy.special import xlogy
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from ...utils.checks import _check_quantile_value, _check_mutual_information_value


def entropy_to_probabilities(entropy: float) -> numpy.ndarray:
    """
    Get a discrete distribution as uniform as possible while achieving
    the desired value of the Shannon entropy.

    Parameters
    ----------
    entropy : float
        Shannon entropy of the resulting discrete distribution.

    Returns
    -------
    probabilities : numpy.ndarray
        Probability vector
    """

    # Shannon entropy is non-negative.
    _check_mutual_information_value(entropy, "entropy")

    n_labels = math.ceil(math.exp(entropy))
    if n_labels == 1:
        return numpy.array([0.0, 1.0])

    # No closed-form solution.
    optimal_p = bisect(lambda x : x * math.log(x) + (1.0 - x) * math.log(1.0 - x) + entropy - (1.0 - x) * math.log(n_labels - 1), 1.0e-15, 1.0 / n_labels, full_output=False)

    probabilities = numpy.full((n_labels,), (1.0 - optimal_p) / (n_labels - 1))
    probabilities[-1] = optimal_p

    return probabilities


class quantized_rv(multi_rv_frozen):
    """
    Frozen quantized distribution with known mutual information.
    """
    
    def __init__(self, base_rv: rv_frozen, 
                 quantiles: list, *args, **kwargs) -> None:
        """
        Create a frozen quantized distribution with known mutual information.

        Parameters
        ----------
        base_rv : scipy.stats._multivariate.multi_rv_frozen
            Base distribution.
        quantiles : array_like
            Quantiles to be used to assign labels.
        """
        
        super().__init__(*args, **kwargs)

        try:
            quantiles = numpy.asarray(quantiles)
        except:
            TypeError("Expected `quantiles` to be convertible to `numpy.ndarray`")

        if len(quantiles) != 1:
            TypeError("Expected `quantiles` to be an 1D array")
            
        _check_quantile_value(quantiles, "quantiles")

        self._quantiles = numpy.sort(quantiles)
        self._dist = base_rv        

    def rvs(self, *args, **kwargs):
        samples = self._dist.rvs(*args, **kwargs)
        labels  = numpy.sum(self._dist.cdf(samples)[:,None] > self._quantiles[None,:], axis=-1)
        
        return (samples, labels)

    @property
    def labels_probabilities(self) -> numpy.ndarray:
        """
        Probabilities of the labels.

        Returns
        -------
        labels_probabilities : numpy.ndarray
            Array of probabilities of the labels
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

        labels_probabilities = self.labels_probabilities
        
        return -numpy.sum(xlogy(labels_probabilities, labels_probabilities))