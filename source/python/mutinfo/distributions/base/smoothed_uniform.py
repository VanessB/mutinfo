import math
import numpy

from scipy.optimize import root_scalar
from scipy.special import lambertw, xlogy
from scipy.stats import uniform, randint
from scipy.stats._multivariate import multi_rv_frozen

from ..tools import BaseMutualInformationTest
from ...utils.special import log_hyperfactorial
from ...utils.checks import _check_dimension_value, _check_mutual_information_value


def _check_inverse_noise_scale_value(
    inverse_noise_scale: float | numpy.ndarray,
    name: str="inverse_noise_scale"
) -> None:
    """
    Checks an inverse smoothing epsilon parameter to be within [0.0; +inf)

    Parameters
    ----------
    inverse_noise_scale : float or array_like
        Inverse noise scale of a smoothed uniform distribution (non-negative).
    """

    if numpy.any(inverse_noise_scale < 0.0):
        raise ValueError(f"Expected `{name}` be non-negative, but got {inverse_noise_scale}")

def _check_noise_scale_value(
    noise_scale: float | numpy.ndarray,
    name: str="noise_scale"
) -> None:
    """
    Checks a noise scale to be within [0.0; +inf)

    Parameters
    ----------
    noise_scale : float or array_like
        Noise scale of a smoothed uniform distribution (non-negative).
    """

    if numpy.any(noise_scale < 0.0):
        raise ValueError(f"Expected `{name}` be non-negative, but got {noise_scale}")

def _check_alphabet_size_value(
    alphabet_size: float | numpy.ndarray,
    name: str="alphabet_size"
) -> None:
    """
    Checks a noise scale to be within [0.0; +inf)

    Parameters
    ----------
    alphabet_size : int or array_like
        Discrete variable support size.
    """

    alphabet_size = numpy.asarray(alphabet_size)
    
    if not numpy.issubdtype(alphabet_size.dtype, numpy.integer):
        raise TypeError(f"Expected `{name}` to be of integer type")

    if numpy.any(alphabet_size < 1):
        raise ValueError(f"Expected `{name}` to be at least 1, but got {alphabet_size}")

def inverse_noise_scale_to_mutual_information(inverse_noise_scale: float | numpy.typing.ArrayLike) -> float | numpy.ndarray:
    """
    Calculate the mutual information between two random variables with a smoothed
    uniform joint distribution defined by the inverse noise scale.

    Parameters
    ----------
    inverse_noise_scale : float or array_like
        Inverse noise scale parameter of a smoothed uniform distribution
        (non-negative).

    Returns
    -------
    mutual_information : float or array_like
        Corresponding mutual information.
    """

    _check_inverse_noise_scale_value(inverse_noise_scale)

    is_scalar = numpy.isscalar(inverse_noise_scale)
    inverse_noise_scale = numpy.asarray(inverse_noise_scale)
    
    mask = inverse_noise_scale > 1.0
    mutual_information = numpy.zeros_like(inverse_noise_scale)
    mutual_information[ mask] = 0.5 / inverse_noise_scale[mask] + numpy.log(inverse_noise_scale[mask])
    mutual_information[~mask] = 0.5 * inverse_noise_scale[~mask]

    return mutual_information.item() if is_scalar else mutual_information 

def mutual_information_to_inverse_noise_scale(mutual_information: float | numpy.ndarray) -> float | numpy.ndarray:
    """
    Calculate the inverse smoothing epsilon given the mutual information
    between two random variables with a smoothed uniform joint distribution.

    Parameters
    ----------
    mutual_information : float or array_like
        Mutual information (lies in [0.0; +inf)).

    Returns
    -------
    inverse_noise_scale : float or array_like
        Corresponding inverse noise scale.
    """

    _check_mutual_information_value(mutual_information)

    is_scalar = numpy.isscalar(mutual_information)
    mutual_information = numpy.asarray(mutual_information)

    mask = mutual_information < 0.5
    inverse_noise_scale = numpy.zeros_like(mutual_information)
    inverse_noise_scale[ mask] = 2.0 * mutual_information[mask]
    inverse_noise_scale[~mask] = -0.5 / numpy.real(lambertw(-0.5 * numpy.exp(-mutual_information[~mask])))

    return inverse_noise_scale.item() if is_scalar else inverse_noise_scale

def inverse_noise_scale_and_alphabet_size_to_mutual_information(
    inverse_noise_scale: float | numpy.typing.ArrayLike,
    alphabet_size: int | numpy.typing.ArrayLike,
) -> float | numpy.ndarray:
    """
    Calculate the mutual information between two random variables with a
    smoothed discrete uniform joint distribution defined by the noise scale
    parameter and alphabet size.

    Parameters
    ----------
    inverse_noise_scale : float or array_like
        Inverse noise scale in a smoothed discrete uniform distribution (non-negative).
    alphabet_size : int or array_like
        Discrete variable support size.

    Returns
    -------
    mutual_information : float or array_like
        Corresponding mutual information.
    """

    _check_alphabet_size_value(alphabet_size)
    _check_inverse_noise_scale_value(inverse_noise_scale)

    inverse_noise_scale_is_scalar = numpy.isscalar(inverse_noise_scale)
    alphabet_size_is_scalar = numpy.isscalar(alphabet_size)
    is_scalar = inverse_noise_scale_is_scalar and alphabet_size_is_scalar

    inverse_noise_scale = numpy.asarray(inverse_noise_scale)
    alphabet_size = numpy.asarray(alphabet_size)

    # Neeeded for using masks.
    if not is_scalar:
        if inverse_noise_scale_is_scalar:
            inverse_noise_scale = numpy.broadcast_to(inverse_noise_scale, alphabet_size.shape)
        elif alphabet_size_is_scalar:
            alphabet_size = numpy.broadcast_to(alphabet_size, inverse_noise_scale.shape)

    # Auxillary values.
    noise_scale = 1.0 / inverse_noise_scale
    floor_noise_scale = numpy.floor(noise_scale).astype(int)
    ceil_noise_scale  = floor_noise_scale + 1
    delta = noise_scale - floor_noise_scale

    zero  = floor_noise_scale == 0.0
    large = floor_noise_scale >= alphabet_size
    small = ~large & ~zero
    mutual_information = numpy.zeros_like(noise_scale)

    # Large noise.
    mutual_information[large] = inverse_noise_scale[large] * (
        (alphabet_size[large] - 1) * numpy.log(alphabet_size[large]) - 2.0 * log_hyperfactorial(alphabet_size[large] - 1) / alphabet_size[large]
    )

    # Small noise.
    mutual_information[small] = numpy.log(alphabet_size[small]) - inverse_noise_scale[small] * (
        2 * log_hyperfactorial(floor_noise_scale[small]) +
        (1 - delta[small]) * (alphabet_size[small] - ceil_noise_scale[small]) * xlogy(floor_noise_scale[small], floor_noise_scale[small]) +
        delta[small] * (alphabet_size[small] - floor_noise_scale[small]) * xlogy(ceil_noise_scale[small], ceil_noise_scale[small])
    ) / alphabet_size[small]

    # Zero noise.
    mutual_information[zero] = numpy.log(alphabet_size[zero])

    return mutual_information.item() if is_scalar else mutual_information

def _scalar_mutual_information_and_alphabet_size_to_inverse_noise_scale(
    mutual_information: float,
    alphabet_size: int,
) -> float:
    """
    Calculate the inverse noise scale given the mutual information between two random
    variables with a smoothed discrete uniform joint distribution.
    (Scalar version)

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies in [0.0; +inf)).
    alphabet_size : int
        Discrete variable support size.

    Returns
    -------
    inverse_noise_scale : float
        Corresponding inverse noise scale.
    """

    _check_alphabet_size_value(alphabet_size)

    log_alphabet_size = math.log(alphabet_size)
    if mutual_information > log_alphabet_size:
        raise ValueError(f"Expected `mutual_information` to be not more than `log(alphabet_size)`, but got {mutual_information} > {log_alphabet_size}")

    if alphabet_size <= 1:
        return 1.0

    # Here mutual_information > 0, so no problems should occur
    lower_bound = mutual_information_to_inverse_noise_scale(mutual_information) / alphabet_size
    upper_bound = 1.0

    # Unfortunatelly, no closed-form expression is available.
    result = root_scalar(
        lambda x : inverse_noise_scale_and_alphabet_size_to_mutual_information(x, alphabet_size) - mutual_information, # Use faster, scalar version.
        bracket=(lower_bound, upper_bound),
    )
    if result.converged:
        return result.root
    else:
        raise ValueError("Unable to find the noise scale.")

_vectorized_mutual_information_and_alphabet_size_to_inverse_noise_scale = numpy.vectorize(_scalar_mutual_information_and_alphabet_size_to_inverse_noise_scale)

def mutual_information_and_alphabet_size_to_inverse_noise_scale(
    mutual_information: float | numpy.typing.ArrayLike,
    alphabet_size: int | numpy.typing.ArrayLike,
) -> float | numpy.ndarray:
    """
    Calculate the inverse noise scale given the mutual information between two
    random variables with a smoothed discrete uniform joint distribution.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies in [0.0; +inf)).
    alphabet_size : int
        Discrete variable support size.

    Returns
    -------
    inverse_noise_scale : float
        Corresponding inverse noise scale.
    """

    _check_mutual_information_value(mutual_information)

    if numpy.isscalar(mutual_information) and numpy.isscalar(alphabet_size):
        inverse_noise_scale = _scalar_mutual_information_and_alphabet_size_to_inverse_noise_scale(mutual_information, alphabet_size)
    else:
        inverse_noise_scale = _vectorized_mutual_information_and_alphabet_size_to_inverse_noise_scale(mutual_information, alphabet_size)
    
    return inverse_noise_scale


class smoothed_uniform(multi_rv_frozen, BaseMutualInformationTest):
    def __init__(
        self,
        inverse_noise_scale: numpy.ndarray,
        *args, **kwargs
    ) -> None:
        """
        Create a multivariate random vector with a smoothed uniform distribution.

        Parameters
        ----------
        inverse_noise_scale : array_like
            1D array of inverse noise scale parameters of the distribution.
        """

        multi_rv_frozen.__init__(self, *args, **kwargs)

        if len(inverse_noise_scale.shape) != 1:
            raise ValueError("`inverse_noise_scale` must be a 1D array")

        _check_inverse_noise_scale_value(inverse_noise_scale)
        self._inverse_noise_scale = inverse_noise_scale

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

        dimensionality = self._inverse_noise_scale.shape[0]
        x = self._dist.rvs(size=(size, dimensionality))
        y = self._dist.rvs(size=(size, dimensionality))

        # Rescale for large noise scale (does not affect mutual information).
        y = x * numpy.minimum(1.0, self._inverse_noise_scale)[None,:] + \
            y / numpy.maximum(1.0, self._inverse_noise_scale)[None,:]
        
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
        return inverse_noise_scale_to_mutual_information(self._inverse_noise_scale)

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


class smoothed_discrete_uniform(multi_rv_frozen, BaseMutualInformationTest):
    def __init__(
        self,
        alphabet_size: int,
        inverse_noise_scale: float,
        normalize: bool=False,
        *args, **kwargs
    ) -> None:
        """
        Create a multivariate random vector with a smoothed discrete uniform distribution.

        Parameters
        ----------
        alphabet_size : int or array_like >= 2
            Discrete variable support size.
        inverse_noise_scale : array_like
            1D array of inverse noise scale parameters of the distribution.
        normalize : bool, optional
            Divide the labels by the total number of possible outcomes.
            Default: False
        """

        multi_rv_frozen.__init__(self, *args, **kwargs)

        if len(alphabet_size.shape) != 1:
            raise ValueError("`alphabet_size` must be a 1D array")
        if len(inverse_noise_scale.shape) != 1:
            raise ValueError("`inverse_noise_scale` must be a 1D array")

        _check_inverse_noise_scale_value(inverse_noise_scale)
        _check_alphabet_size_value(alphabet_size)

        self._inverse_noise_scale = inverse_noise_scale
        self._alphabet_size = alphabet_size
        self.normalize = normalize

        self.x_dist = randint
        self.z_dist = uniform()

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

        dimensionality = self._inverse_noise_scale.shape[0]
        x = numpy.stack([self.x_dist.rvs(low=0, high=high, size=size) for high in self._alphabet_size], axis=-1)
        y = x + self.z_dist.rvs(size=(size, dimensionality)) / self._inverse_noise_scale[None,:]
        
        if self.normalize:
            x = x / numpy.maximum(1, self._alphabet_size - 1)
            y = y / numpy.maximum(1, self._alphabet_size - 1 + 1.0 / self._inverse_noise_scale)

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
        return inverse_noise_scale_and_alphabet_size_to_mutual_information(self._inverse_noise_scale, self._alphabet_size)

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