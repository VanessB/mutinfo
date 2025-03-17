import math
import numpy

from scipy.optimize import bisect
from scipy.special import xlogy
from scipy.stats._distn_infrastructure import rv_frozen, rv_sample
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
        Probability vector.
    """

    # Shannon entropy is non-negative.
    _check_mutual_information_value(entropy, "entropy")

    n_labels = math.ceil(math.exp(entropy))
    if n_labels == 1:
        return numpy.array([1.0])

    function = lambda x : xlogy(x, x) + xlogy(1.0 - x, 1.0 - x) + entropy - (1.0 - x) * math.log(n_labels - 1)
    left_end  = 0.0
    right_end = 1.0 / n_labels

    # Corner case: close to zero if function(right_end) is non-negative.
    if function(right_end) >= 0.0:
        return numpy.full((n_labels,), 1.0 / n_labels)

    # No closed-form solution, using bisection method.
    optimal_p = bisect(function, left_end, right_end, full_output=False)

    # Corner case.
    if optimal_p == 0.0:
        return numpy.full((n_labels-1,), 1.0 / (n_labels - 1))

    probabilities = numpy.full((n_labels,), (1.0 - optimal_p) / (n_labels - 1))
    probabilities[-1] = optimal_p

    return probabilities


class splitted_rv_sample(rv_sample):
    """
    Frozen discrete distribution with known mutual information.
    """
    
    def __init__(
        self,
        values: tuple[numpy.ndarray, numpy.ndarray],
        split_dim: int=1,
        *args,
        **kwargs
    ) -> None:
        """
        Create a frozen discrete distribution with known mutual information.

        Parameters
        ----------
        values : tuple of two array_like, optional
            
        split_dim : int
            Dimension used to split the vector.
        """

        super().__init__(values=values, *args, **kwds)

        if split_dim < 1:
            raise ValueError(f"Expected `split_dim` to be 1 or greater, but got {split_dim}")

        if len(values[1].shape) < split_dim:
            raise ValueError(
                f"Expected the dimensionality of the probability tensor to be breater then `split_dim`, but got {len(values[1].shape)} < {split_dim}"
            )

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        
        x_y_probabilities = self._ctor_param["values"][1]
        x_probabilities   = x_y_probabilities.sum(axis=range(self.split_dim))
        y_probabilities   = x_y_probabilities.sum(axis=range(self.split_dim, len(x_y_probabilities.shape)))
        
        return xlogy(x_y_probabilities, x_y_probabilities).sum() \
            - xlogy(x_probabilities, x_probabilities).sum() \
            - xlogy(y_probabilities, y_probabilities).sum()


class quantized_rv(multi_rv_frozen):
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

        self.normalize = normalize

    def rvs(self, *args, **kwargs):
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