import numpy
from scipy.special import lambertw
from scipy.stats import uniform
from scipy.stats._multivariate import multi_rv_frozen

from ...utils.checks import _check_dimension_value, _check_mutual_information_value


def _check_inverse_smoothing_epsilon_value(inverse_smoothing_epsilon: float, name: str="inverse_smoothing_epsilon") -> None:
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

def inverse_smoothing_epsilon_to_mutual_information(inverse_smoothing_epsilon: float) -> float:
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

def mutual_information_to_inverse_smoothing_epsilon(mutual_information: float) -> float:
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
    def __init__(self, inverse_smoothing_epsilon: numpy.ndarray,
                 X_dimension: int=None, Y_dimension: int=None, *args, **kwargs) -> None:
        """
        Create a multivariate random vector with
        a smoothed uniform distribution.

        Parameters
        ----------
        inverse_smoothing_epsilon : array_like
            1D array of inverse smoothing epsilon parameters of the distribution.
        X_dimension : int, optional
            Dimensionality of the first vector.
        Y_dimension : int, optional
            Dimensionality of the second vector.
        """

        super().__init__(*args, **kwargs)

        if not X_dimension is None:
            _check_dimension_value(X_dimension, "X_dimension")
        if not Y_dimension is None:
            _check_dimension_value(Y_dimension, "Y_dimension")

        if len(inverse_smoothing_epsilon.shape) != 1:
            raise ValueError("`inverse_smoothing_epsilon` must be a 1D array")

        _check_inverse_smoothing_epsilon_value(inverse_smoothing_epsilon)
        self._inverse_smoothing_epsilon = inverse_smoothing_epsilon

        min_dimension = self._inverse_smoothing_epsilon.shape[0]
        self._X_dimension = min_dimension if X_dimension is None else X_dimension
        self._Y_dimension = min_dimension if Y_dimension is None else Y_dimension

        if (self._X_dimension != min_dimension and self._Y_dimension != min_dimension) or \
           (self._X_dimension < min_dimension or self._Y_dimension < min_dimension):
            raise ValueError("Dimensions of vectors can not be deduced; try checking the shape of `inverse_smoothing_epsilon` and the values of `X/Y_dimension`")

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
        x_y : numpy.ndarray
            Random sampling.
        """
        
        x = self._dist.rvs(size=(size, self._X_dimension))
        y = self._dist.rvs(size=(size, self._Y_dimension))

        min_dimension = min(self._X_dimension, self._Y_dimension)

        # Rescale for large smoothing epsilons (does not affect mutual information).
        y[...,-min_dimension:] = x[...,:min_dimension]  * numpy.minimum(1.0, 0.5 * self._inverse_smoothing_epsilon)[None,:] + \
                                 y[...,-min_dimension:] / numpy.maximum(1.0, 0.5 * self._inverse_smoothing_epsilon)[None,:]
        return x, y

    @property
    def componentwise_mutual_information(self) -> numpy.ndarray:
        """
        Componentwise mutual information.

        Returns
        -------
        componentwise_mutual_information : np.array
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
            Mutual information
        """
        return numpy.sum(self.componentwise_mutual_information)