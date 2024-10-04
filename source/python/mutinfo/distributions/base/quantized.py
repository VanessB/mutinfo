import math
import numpy

from scipy.optimize import bisect
from scipy.special import xlogy
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from ...utils.checks import _check_dimension_value, _check_mutual_information_value, _check_quantile_value


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
    
    def __init__(self, base_rv: rv_frozen, quantiles: list,
                 X_dimension: int=1, Y_dimension: int=None,
                 *args, **kwargs) -> None:
        """
        Create a frozen quantized distribution with known mutual information.

        Parameters
        ----------
        base_rv : scipy.stats._multivariate.multi_rv_frozen
            Base distribution.
        quantiles : array_like
            Quantiles to be used to assign labels.
        X_dimension : int, optional
            Dimensionality of the first vector.
        Y_dimension : int, optional
            Dimensionality of the second vector.
        """
        
        super().__init__(*args, **kwargs)

        _check_dimension_value(X_dimension, "X_dimension")
        if not Y_dimension is None:
            _check_dimension_value(Y_dimension, "Y_dimension")
        else:
            Y_dimension = X_dimension

        try:
            quantiles = numpy.asarray(quantiles)
        except:
            TypeError("Expected `quantiles` to be convertible to `numpy.ndarray`")

        if len(quantiles) != 1:
            TypeError("Expected `quantiles` to be an 1D array")
            
        _check_quantile_value(quantiles, "quantiles")

        self._X_dimension = X_dimension
        self._Y_dimension = Y_dimension

        self._quantiles = numpy.sort(quantiles)
        self._dist = base_rv        

    def rvs(self, size: int=1, *args, **kwargs):
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x_y : numpy.ndarray
            Random sampling.
        """

        min_dimension = min(self._X_dimension, self._Y_dimension)
        
        x = self._dist.rvs(size=(size, self._X_dimension), *args, **kwargs)

        y = numpy.empty((size, self._Y_dimension), dtype=numpy.int64)
        y[:,:min_dimension] = numpy.sum(self._dist.cdf(x[:,:min_dimension])[...,None] > self._quantiles[None,None,:], axis=-1)
        if min_dimension < self._Y_dimension:
            y[:,min_dimension:self._Y_dimension] = numpy.random.choice(self._quantiles.shape[0] + 1, (size, self._Y_dimension - min_dimension), p=self.label_probabilities)
        
        return x, y

    @property
    def label_probabilities(self) -> numpy.ndarray:
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

        label_probabilities = self.label_probabilities
        
        return -numpy.sum(xlogy(label_probabilities, label_probabilities))