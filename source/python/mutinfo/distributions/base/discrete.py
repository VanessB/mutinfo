import math
import numpy

from scipy.optimize import bisect, root_scalar
from scipy.special import xlogy
from scipy.stats import randint, uniform
from scipy.stats._distn_infrastructure import rv_frozen, rv_discrete_frozen, rv_sample
from scipy.stats._multivariate import multi_rv_frozen

from ...utils.checks import _check_dimension_value, _check_mutual_information_value, _check_probability_value


def _entropy_to_length_and_remainder(entropy: float) -> tuple[int, float]:
    # Shannon entropy is non-negative.
    _check_mutual_information_value(entropy, "entropy")

    n_labels = math.ceil(math.exp(entropy))
    if n_labels == 1:
        return 1, 1.0

    function  = lambda x : xlogy(x, x) + xlogy(1.0 - x, 1.0 - x) + entropy - (1.0 - x) * math.log(n_labels - 1)
    left_end  = 0.0
    right_end = 1.0 / n_labels

    # Corner case: close to zero if function(right_end) is non-negative.
    if function(right_end) >= 0.0:
        return n_labels, 1.0 / n_labels

    # No closed-form solution, using bisection method.
    optimal_p = bisect(function, left_end, right_end, full_output=False)

    # Corner case.
    if optimal_p == 0.0:
        return n_labels-1, 1.0 / (n_labels-1)

    return n_labels, optimal_p

def _length_and_remainder_to_probabilities(length: int, residual: float) -> numpy.ndarray:
    if length == 1:
        return numpy.array([1.0])
    
    probabilities = numpy.full((length,), (1.0 - residual) / (length - 1))
    probabilities[-1] = residual

    return probabilities

def entropy_to_probabilities(entropy: float) -> numpy.ndarray:
    """
    Get a discrete distribution which is as uniform as possible while achieving
    the desired value of Shannon entropy.

    Parameters
    ----------
    entropy : float
        Shannon entropy of the resulting discrete distribution.

    Returns
    -------
    probabilities : numpy.ndarray
        Probability vector.
    """

    return _length_and_remainder_to_probabilities(*_entropy_to_length_and_remainder(entropy))


def _scalar_mutual_information_to_reroll_probability(
    mutual_information: float,
    alphabet_size: int,
) -> float:
    """
    Calculate reroll probability in a discrete symmetric noisy channel with
    uniformly distributed input corresponding to a given value of
    mutual information.
    (scalar version)

    Parameters
    ----------
    mutual_information : float or array_like
        Mutual information.
    alphabet_size : int or array_like
        Alphabet size.

    Returns
    -------
    reroll_probability : float
        Reroll probability.
    """

    lower_bound = 0.0
    upper_bound = 1.0 - mutual_information / math.log(alphabet_size)

    # Unfortunatelly, no closed-form expression is available.
    result = root_scalar(
        lambda x : reroll_probability_to_mutual_information(x, alphabet_size) - mutual_information,
        bracket=(lower_bound, upper_bound),
    )
    if result.converged:
        return result.root
    else:
        raise ValueError("Unable to find the reroll probability.")

_vectorized_mutual_information_to_reroll_probability = numpy.vectorize(_scalar_mutual_information_to_reroll_probability)


def reroll_probability_to_mutual_information(
    reroll_probability: float | numpy.ndarray,
    alphabet_size: int | numpy.ndarray,
) -> float:
    """
    Calculate mutual information in a discrete symmetric noisy channel with
    uniformly distributed input and given reroll probability.

    Parameters
    ----------
    reroll_probability : float or array_like
        Reroll probability.
    alphabet_size : int or array_like
        Alphabet size.

    Returns
    -------
    mutual_information : float
        Mutual information.
    """

    success_probability = 1.0 - reroll_probability * (1.0 - 1.0 / alphabet_size)
    
    entropy = numpy.log(alphabet_size)
    negative_conditional_entropy = xlogy(success_probability, success_probability) + \
            xlogy(1.0 - success_probability, reroll_probability / alphabet_size)

    return numpy.maximum(0.0, entropy + negative_conditional_entropy)


def mutual_information_to_reroll_probability(
    mutual_information: float | numpy.ndarray,
    alphabet_size: int | numpy.ndarray,
) -> float:
    """
    Calculate reroll probability in a discrete symmetric noisy channel with
    uniformly distributed input corresponding to a given value of
    mutual information.

    Parameters
    ----------
    mutual_information : float or array_like
        Mutual information.
    alphabet_size : int or array_like
        Alphabet size.

    Returns
    -------
    reroll_probability : float
        Reroll probability.
    """

    return _vectorized_mutual_information_to_reroll_probability(mutual_information, alphabet_size)


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
        values : tuple of two array_like.
            Values and corresponding probabilities (joint PMF).
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

    def rvs(self, size: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        An adapter to a SciPy `rv_sample.rvs`.
        """
        
        x_y = super().rvs(size=size)
        
        return x_y[:,self.split_dim:], x_y[:,:self.split_dim]

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


class symmetric_noisy_channel(rv_discrete_frozen):
    """
    Frozen discrete joint distribution of a symmetric noisy channel.
    """

    def __init__(
        self,
        values: tuple[numpy.ndarray, numpy.ndarray],
        reroll_probability: float=0.0,
        permutation: numpy.ndarray=None,
    ) -> None:
        """
        Create a discrete joint distribution of a symmetric noisy channel.

        Parameters
        ----------
        values : tuple of two array_like.
            Values and corresponding probabilities of the input (marginal PMF).
        reroll_probability : float, optional
            Probability of transmitting random noise instead of input.
            Not the probability of an error!
        permutation : array_like, optional
            Permutation to be applied to Y. No permutation if `None`.
        """
        
        self._dist = rv_sample(values=values)
        self._reroll_dist = uniform()
        self._reroll_outcome_dist = randint(low=0, high=len(self.values[0]))

        self.reroll_probability = reroll_probability
        self.permutation = permutation

    def rvs(self, size: int) -> tuple[numpy.ndarray, numpy.ndarray]:
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
        
        x = self._dist.rvs(size=size)
        y = x.copy()

        if self.reroll_probability > 0.0:
            rerolled = self._reroll_dist.rvs(size=size) <= self.reroll_probability
            y[rerolled] = self.values[0][self._reroll_outcome_dist.rvs(size=rerolled.sum())]

        if not (self.permutation is None):
            y = self.permutation[y]
        
        return x, y

    @property
    def values(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Values and corresponding probabilities of the input (marginal PMF).
        """

        return self._dist._ctor_param["values"]

    #@property
    #def error_probability(self) -> float:
    #    """
    #    Probability of an error (random and equiprobable switch to any other value).
    #    """
    #    return self._error_probability

    #@error_probability.setter
    #def error_probability(self, error_probability: float) -> None:
    #    if error_probability > (1.0 - 1.0 / len(self.values[0])):
    #        raise ValueError("The value of `error_probability` can not exceed `(1 - 1/n)`, where `n` is the size of the alphabet.")

    #    self._error_probability = error_probability

    #@property
    #def reroll_probability(self) -> float:
    #    """
    #    Reroll probability (probability of transmitting random noise innstead of input).
    #    """

    #    n_values = len(self.values[0])

    #    # TODO: this is clearly bad if `self.error_probability` > (1 - 1/n_values)
    #    return self.error_probability * n_values / (n_values - 1)

    @property
    def reroll_probability(self) -> float:
        """
        Reroll probability (probability of transmitting random noise instead of input).

        Returns
        -------
        reroll_probability : float
            Reroll probability
        """

        return self._reroll_probability

    @reroll_probability.setter
    def reroll_probability(self, reroll_probability: float) -> None:
        _check_probability_value(reroll_probability, "reroll_probability")
        self._reroll_probability = reroll_probability

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """

        n_values = len(self.values[0])
        
        x_probabilities = self.values[1]
        y_probabilities = (1.0 - self.reroll_probability) * x_probabilities + self.reroll_probability / n_values
        y_conditional_probabilities = numpy.eye(n_values) * (1.0 - self.reroll_probability) + \
                np.ones((n_values, n_values)) * self.reroll_probability / n_values

        y_entropy = -xlogy(y_probabilities, y_probabilities).sum()
        y_conditional_entropy = -(x_probabilities * xlogy(y_conditional_probabilities, y_conditional_probabilities).sum(axis=0)).sum()
        
        return y_entropy - y_conditional_entropy


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
            
        _check_probability_value(quantiles, "quantiles")

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