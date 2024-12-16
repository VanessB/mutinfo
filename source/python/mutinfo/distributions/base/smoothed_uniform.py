import numpy

from scipy.special import lambertw
from scipy.stats import uniform
from scipy.stats._multivariate import multi_rv_frozen

from ...utils.checks import _check_dimension_value, _check_mutual_information_value


def _check_inverse_smoothing_epsilon_value(
    inverse_smoothing_epsilon: float | numpy.ndarray,
    name: str="inverse_smoothing_epsilon"
) -> None:
    """
    Checks an inverse smoothing epsilon parameter to be within [0.0; +inf)

    Parameters
    ----------
    inverse_smoothing_epsilon : float or array_like
        Inverse smoothing epsilon parameter of a smoothed uniform distribution
        (non-negative).
    """

    if numpy.any(inverse_smoothing_epsilon < 0.0):
        raise ValueError(f"Expected `{inverse_smoothing_epsilon}` be non-negative")

def inverse_smoothing_epsilon_to_mutual_information(inverse_smoothing_epsilon: float | numpy.ndarray) -> float | numpy.ndarray:
    """
    Calculate the mutual information between two random variables with a smoothed
    uniform joint distribution defined by the inverse smoothing epsilon.

    Parameters
    ----------
    inverse_smoothing_epsilon : float or array_like
        Inverse smoothing epsilon parameter of a smoothed uniform distribution
        (strictly positive).

    Returns
    -------
    mutual_information : float or array_like
        Corresponding mutual information.
    """

    _check_inverse_smoothing_epsilon_value(inverse_smoothing_epsilon)

    is_float = isinstance(inverse_smoothing_epsilon, float)
    inverse_smoothing_epsilon = numpy.asarray(inverse_smoothing_epsilon)
    
    mask = inverse_smoothing_epsilon > 2.0
    mutual_information = numpy.zeros_like(inverse_smoothing_epsilon)
    mutual_information[ mask] = 1.0 / inverse_smoothing_epsilon[mask] + numpy.log(0.5 * inverse_smoothing_epsilon[mask])
    mutual_information[~mask] = 0.25 * inverse_smoothing_epsilon[~mask]

    return mutual_information.item() if is_float else mutual_information 

def mutual_information_to_inverse_smoothing_epsilon(mutual_information: float | numpy.ndarray) -> float | numpy.ndarray:
    """
    Calculate the inverse smoothing epsilon given the mutual information
    between two random variables with a smoothed uniform joint distribution.

    Parameters
    ----------
    mutual_information : float or array_like
        Mutual information (lies in [0.0; +inf)).

    Returns
    -------
    inverse_smoothing_epsilon : float or array_like
        Corresponding inverse smoothing epsilon.
    """

    _check_mutual_information_value(mutual_information)

    is_float = isinstance(mutual_information, float)
    mutual_information = numpy.asarray(mutual_information)

    mask = mutual_information < 0.5
    inverse_smoothing_epsilon = numpy.zeros_like(mutual_information)
    inverse_smoothing_epsilon[ mask] = 4.0 * mutual_information[mask]
    inverse_smoothing_epsilon[~mask] = -1.0 / numpy.real(lambertw(-0.5 * numpy.exp(-mutual_information[~mask])))

    return inverse_smoothing_epsilon.item() if is_float else inverse_smoothing_epsilon 


class smoothed_uniform(multi_rv_frozen):
    def __init__(
        self,
        inverse_smoothing_epsilon: numpy.ndarray,
        *args, **kwargs
    ) -> None:
        """
        Create a multivariate random vector with
        a smoothed uniform distribution.

        Parameters
        ----------
        inverse_smoothing_epsilon : array_like
            1D array of inverse smoothing epsilon parameters of the distribution.
        """

        super().__init__(*args, **kwargs)

        if len(inverse_smoothing_epsilon.shape) != 1:
            raise ValueError("`inverse_smoothing_epsilon` must be a 1D array")

        _check_inverse_smoothing_epsilon_value(inverse_smoothing_epsilon)
        self._inverse_smoothing_epsilon = inverse_smoothing_epsilon

        self._dist = uniform()

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

        dimensionality = self._inverse_smoothing_epsilon.shape[0]
        x = self._dist.rvs(size=(size, dimensionality))
        y = self._dist.rvs(size=(size, dimensionality))

        # Rescale for large smoothing epsilons (does not affect mutual information).
        y = x * numpy.minimum(1.0, 0.5 * self._inverse_smoothing_epsilon)[None,:] + \
            y / numpy.maximum(1.0, 0.5 * self._inverse_smoothing_epsilon)[None,:]
        
        return x, y

    @property
    def componentwise_mutual_information(self) -> numpy.ndarray:
        """
        Componentwise mutual information.

        Returns
        -------
        componentwise_mutual_information : numpy.ndarray
            Componentwise mutual information
        """
        return inverse_smoothing_epsilon_to_mutual_information(self._inverse_smoothing_epsilon)

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        return numpy.sum(self.componentwise_mutual_information)