import numpy
import math
from scipy.optimize import root_scalar
from scipy.special import digamma
from scipy.stats import gamma, expon
from scipy.stats._multivariate import multi_rv_frozen

from ...utils.checks import _check_dimension_value, _check_mutual_information_value


_EPS = 1.0e-6

def _check_inverse_shape_parameter_value(inverse_shape_parameter: float, name: str="inverse_shape_parameter") -> None:
    """
    Checks inverse shape parameter to be within [0.0; +inf)

    Parameters
    ----------
    inverse_shape_parameter : float or array_like
        Inverse shape parameter of a gamma-exponential distribution
        (non-negative).
    name : str, optional
        Name of the variable to be checked.
        Default is "inverse_shape_parameter"
    """

    if numpy.any(inverse_shape_parameter < 0.0):
        raise ValueError(f"Expected `{name}` to be non-negative")

def _scalar_inverse_shape_parameter_to_mutual_information(inverse_shape_parameter: float) -> float:
    """
    Calculate the mutual information between two random variables with a
    gamma-exponential joint distribution defined by the inverse shape parameter.
    (Scalar version)

    Parameters
    ----------
    shape_parameter : float
        Shape parameter of a gamma-exponential distribution
        (strictly positive).

    Returns
    -------
    mutual_information : float
        Corresponding mutual information.
    """

    if inverse_shape_parameter < 2.0 * _EPS:
        return 0.5 * inverse_shape_parameter
    else:
        return digamma(1.0 / inverse_shape_parameter) + inverse_shape_parameter + math.log(inverse_shape_parameter)

def _scalar_mutual_information_to_inverse_shape_parameter(mutual_information: float) -> float:
    """
    Calculate the inverse shape parameter given the mutual information
    between two random variables with a gamma-exponential joint distribution.
    (Scalar version)

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies in [0.0; +inf)).

    Returns
    -------
    inverse_shape_parameter : float
        Corresponding inverse correlation coefficient.
    """

    lower_bound = math.exp(mutual_information + numpy.euler_gamma)
    upper_bound = min(2.0 * mutual_information, math.exp(mutual_information) - 1.0)

    # Unfortunatelly, no closed-form expression is available.
    result = root_scalar(
        lambda x : _scalar_inverse_shape_parameter_to_mutual_information(x) - mutual_information,
        bracket=(lower_bound, upper_bound),
    )
    if result.converged:
        return result.root
    else:
        raise ValueError("Unable to find the inverse shape parameter")

_vectorized_mutual_information_to_inverse_shape_parameter = numpy.vectorize(_scalar_mutual_information_to_inverse_shape_parameter)

def inverse_shape_parameter_to_mutual_information(inverse_shape_parameter: float) -> float:
    """
    Calculate the mutual information between two random variables with a
    gamma-exponential joint distribution defined by the inverse shape parameter.

    Parameters
    ----------
    shape_parameter : array_like
        Shape parameter of a gamma-exponential distribution
        (strictly positive).

    Returns
    -------
    mutual_information : array_like
        Corresponding mutual information.
    """

    _check_inverse_shape_parameter_value(inverse_shape_parameter)

    is_float = isinstance(inverse_shape_parameter, float)
    inverse_shape_parameter = numpy.asarray(inverse_shape_parameter)

    mask = inverse_shape_parameter < 2.0 * _EPS
    mutual_information = numpy.zeros_like(inverse_shape_parameter)
    mutual_information[mask]  = 0.5 * inverse_shape_parameter[mask]
    mutual_information[~mask] = digamma(1.0 / inverse_shape_parameter[~mask]) + \
                                inverse_shape_parameter[~mask] + numpy.log(inverse_shape_parameter[~mask])

    return mutual_information.item() if is_float else mutual_information

def mutual_information_to_inverse_shape_parameter(mutual_information: float) -> float:
    """
    Calculate the inverse shape parameter given the mutual information
    between two random variables with a gamma-exponential joint distribution.

    Parameters
    ----------
    mutual_information : float or array_like
        Mutual information (lies in [0.0; +inf)).

    Returns
    -------
    inverse_shape_parameter : float or array_like
        Corresponding inverse correlation coefficient.
    """

    _check_mutual_information_value(mutual_information)

    if isinstance(mutual_information, float):
        inverse_shape_parameter = _scalar_mutual_information_to_inverse_shape_parameter(mutual_information)
    else:
        inverse_shape_parameter = _vectorized_mutual_information_to_inverse_shape_parameter(mutual_information)
    
    return inverse_shape_parameter


class gamma_exponential(multi_rv_frozen):
    def __init__(self, inverse_shape_parameter: numpy.ndarray,
                 X_dimension: int=None, Y_dimension: int=None, *args, **kwargs) -> None:
        """
        Create a multivariate random vector with
        a gamma-exponential distribution.

        Parameters
        ----------
        inverse_shape_parameter : array_like
            1D array of inverse shape parameters of the distribution.
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

        if len(inverse_shape_parameter.shape) != 1:
            raise ValueError("`inverse_shape_parameter` must be a 1D array")

        _check_inverse_shape_parameter_value(inverse_shape_parameter)
        self._inverse_shape_parameter = inverse_shape_parameter

        min_dimension = self._inverse_shape_parameter.shape[0]
        self._X_dimension = min_dimension if X_dimension is None else X_dimension
        self._Y_dimension = min_dimension if Y_dimension is None else Y_dimension

        if (self._X_dimension != min_dimension and self._Y_dimension != min_dimension) or \
           (self._X_dimension < min_dimension or self._Y_dimension < min_dimension):
            raise ValueError("Dimensions of vectors can not be deduced; try checking the shape of `inverse_shape_parameter` and the values of `X/Y_dimension`")

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

        x = numpy.stack([numpy.ones(size) if k <= _EPS else gamma.rvs(a=1.0/k, loc=1.0e-12, scale=k, size=(size,)) for k in self._inverse_shape_parameter], axis=1)
        y = expon.rvs(size=(size, self._Y_dimension))

        min_dimension = min(self._X_dimension, self._Y_dimension)

        # Numerically stable, mutual information is not alternated.
        # TODO: make new random variable or rename this to account for log
        x = numpy.log(x)
        y = numpy.log(y)
        y[:,-min_dimension:] = y[:,-min_dimension:] - x[:,-min_dimension:]
        
        #y[:,-min_dimension:] = y[:,-min_dimension:] / x[:,-min_dimension:]
        
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
        return inverse_shape_parameter_to_mutual_information(self._inverse_shape_parameter)

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