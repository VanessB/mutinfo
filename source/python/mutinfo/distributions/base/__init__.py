import math
import numpy

from scipy.special import ndtr, ndtri
from scipy.stats import ortho_group
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from . import discrete
from . import gamma_exponential
from . import normal
from . import smoothed_uniform
from . import student
from . import mixture
from .. import tools

from ...utils.checks import _check_dimension_value, _check_mutual_information_value


def _sample_from_simplex(dimensionality: int) -> numpy.ndarray:
    """
    Obtain a sample from the uniform distribution on a multidimensional simplex.

    Parameters
    ----------
    dimensionality : int
        Dimensionality of the simplex.

    Returns
    -------
    sample ; numpy.ndarray
        One sample from the uniform distribution on a multidimensional simplex.
    """
    
    result = numpy.random.exponential(scale=1.0, size=dimensionality)
    return result / numpy.sum(result)


def _distribute_mutual_information(
    mutual_information: float,
    dimensionality: int,
    uniform: bool=True
) -> numpy.ndarray:
    """
    Uniformly or randomly distribute mutual information along dimensions.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality : int
        Number of the dimensions.
    uniform : bool
        Distribute uniformly.

    Returns
    -------
    componentwise_mutual_information : numpy.ndarray
        1D array of componentwise mutual information values.
    """

    _check_dimension_value(dimensionality)

    if uniform:
        componentwise_mutual_information = mutual_information * numpy.ones(dimensionality) / dimensionality
    else:
        componentwise_mutual_information = mutual_information * _sample_from_simplex(dimensionality)

    return componentwise_mutual_information
        


def _generate_cov_via_tridiagonal(
    mutual_information: float,
    dimensionality: int | tuple[int, int],
    randomize_interactions: bool=True,
    shuffle_interactions: bool=True
) -> normal.CovViaTridiagonal:
    """
    Create a covariance matrix for a correlated multivariate normal distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality: int or tuple[int, int],
        Dimensionality of the vectors.
    randomize_interactions : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.
    shuffle_interactions : bool, optional
        Use orthogonal matrices to randomize off-diagonal block of the
        covariation matrix (mutual information stays fixed).

    Returns
    -------
    random_variable : normal.CovViaTridiagonal
        An instance of normal.CovViaTridiagonal
        with the provided value of the mutual information.
    """

    if isinstance(dimensionality, int):
        dimensionality = (dimensionality, dimensionality)
    elif not isinstance(dimensionality, tuple):
        raise ValueError("Expected `dimensionality` to be of type `int` or `tuple[int, int]`")
    
    min_dim = min(dimensionality)
    componentwise_mutual_information = _distribute_mutual_information(mutual_information, min_dim, not randomize_interactions)
    correlation_coefficient = normal.mutual_information_to_correlation(componentwise_mutual_information)

    if shuffle_interactions:
        X_orthogonal_matrix = None if dimensionality[0] == 1 else ortho_group.rvs(dimensionality[0])
        Y_orthogonal_matrix = None if dimensionality[1] == 1 else ortho_group.rvs(dimensionality[1])
    else:
        X_orthogonal_matrix = None
        Y_orthogonal_matrix = None

    return normal.CovViaTridiagonal(correlation_coefficient, X_orthogonal_matrix, Y_orthogonal_matrix)


def CorrelatedNormal(*args, **kwargs) -> normal.correlated_multivariate_normal:
    """
    Create a multivariate correlated normal distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality: int or tuple[int, int],
        Dimensionality of the vectors.
    randomize_interactions : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.
    shuffle_interactions : bool, optional
        Use orthogonal matrices to randomize off-diagonal block of the
        covariation matrix (mutual information stays fixed).

    Returns
    -------
    random_variable : normal.correlated_multivariate_normal
        An instance of normal.correlated_multivariate_normal
        with the provided value of the mutual information.
    """

    covariance = _generate_cov_via_tridiagonal(*args, **kwargs)
    return normal.correlated_multivariate_normal(covariance)


def CorrelatedUniform(*args, **kwargs) -> tools.mapped_multi_rv_frozen:
    """
    Create a multivariate correlated uniform distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality: int or tuple[int, int],
        Dimensionality of the vectors.
    randomize_correlation : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.
    shuffle_interactions : bool, optional
        Use orthogonal matrices to randomize off-diagonal block of the
        covariation matrix (mutual information stays fixed).

    Returns
    -------
    random_variable : mapped.mapped_multi_rv_frozen
        An instance of mapped.mapped_multi_rv_frozen
        with the provided value of the mutual information
        and ndtr (normal to uniform) mapping.
    """

    # Use Gaussian CDF to acquire the uniform distribution.
    return tools.mapped_multi_rv_frozen(CorrelatedNormal(*args, **kwargs), lambda x, y: (ndtr(x), ndtr(y)), lambda x, y: (ndtri(x), ndtri(y)))


def CorrelatedStudent(
    mutual_information: float,
    dimensionality: int | tuple[int, int],
    degrees_of_freedom: int,
    randomize_interactions: bool=True,
    shuffle_interactions: bool=True
) -> student.correlated_multivariate_student:
    """
    Create a multivariate correlated Student's distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality: int or tuple[int, int],
        Dimensionality of the vectors.
    degrees_of_freedom : int
        Number of degrees of freedom.
    randomize_interactions : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.
    shuffle_interactions : bool, optional
        Use orthogonal matrices to randomize off-diagonal block of the
        covariation matrix (mutual information stays fixed).

    Returns
    -------
    random_variable : student.correlated_multivariate_student
        An instance of student.correlated_multivariate_student
        with the provided value of the mutual information.
    """

    if isinstance(dimensionality, int):
        dimensionality = (dimensionality, dimensionality)
    elif not isinstance(dimensionality, tuple):
        raise ValueError("Expected `dimensionality` to be of type `int` or `tuple[int, int]`")

    correction_term = student.mutual_information_correction_term(
        dimensionality[0], dimensionality[1], degrees_of_freedom
    )

    correlation_mutual_information = mutual_information - correction_term
    if correlation_mutual_information < 0.0:
        raise ValueError(f"Impossible to create a correlated Student's distribution with the desired mutual information. The latter should be at least {correction_term:.2f} nat")

    covariance = _generate_cov_via_tridiagonal(
        correlation_mutual_information,
        dimensionality,
        randomize_interactions,
        shuffle_interactions,
    )
    return student.correlated_multivariate_student(covariance, degrees_of_freedom)


def LogGammaExponential(
    mutual_information: float,
    dimensionality: int,
    randomize_interactions: bool=True
) -> gamma_exponential.log_gamma_exponential:
    """
    Create a multivariate log-gamma-exponential distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality : int
        Dimensionality of the vectors.
    randomize_interactions : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.

    Returns
    -------
    random_variable : gamma_exponential.log_gamma_exponential
        An instance of gamma_exponential.log_gamma_exponential
        with the provided value of the mutual information.
    """

    componentwise_mutual_information = _distribute_mutual_information(mutual_information, dimensionality, not randomize_interactions)
    inverse_shape_parameter = gamma_exponential.mutual_information_to_inverse_shape_parameter(componentwise_mutual_information)
    
    return gamma_exponential.log_gamma_exponential(inverse_shape_parameter)


def UniformlyQuantized(
    mutual_information: float,
    dimensionality: int,
    base_rv: rv_frozen,
    normalize: bool=False,
    randomize_interactions: bool=False
) -> discrete.quantized_rv:
    """
    Create a two-dimensional mixed-type distribution
    given the value of the mutual information between the components.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality : int
        Dimensionality of the vectors.
    base_rv : scipy.stats._multivariate.multi_rv_frozen
        Base univariate distribution for the first component.
    normalize : bool, optional
        Divide the labels by the total number of possible outcomes.
        Default: False
    randomize_interactions : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.

    Returns
    -------
    random_variable : quantized.quantized_rv
        An instance of quantized.quantized_rv
        with the provided value of the mutual information.
    """

    if randomize_interactions:
        raise NotImplementedError("Interaction randomization is not implemented for `UniformlyQuantized` yet.")

    probabilities = discrete.entropy_to_probabilities(mutual_information / dimensionality)
    quantiles = numpy.cumsum(probabilities)[:-1]

    return tools.stacked_multi_rv_frozen(discrete.quantized_rv(base_rv, quantiles, normalize), dimensionality)


def SmoothedUniform(
    mutual_information: float,
    dimensionality: int,
    randomize_interactions: bool=True
) -> smoothed_uniform.smoothed_uniform:
    """
    Create a multivariate smoothed uniform distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality : int
        Dimensionality of the vectors.
    randomize_interactions : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.

    Returns
    -------
    random_variable : smoothed_uniform.smoothed_uniform
        An instance of smoothed_uniform.smoothed_uniform
        with the provided value of the mutual information.
    """

    componentwise_mutual_information = _distribute_mutual_information(mutual_information, dimensionality, not randomize_interactions)
    inverse_smoothing_epsilon = smoothed_uniform.mutual_information_to_inverse_smoothing_epsilon(componentwise_mutual_information)
    
    return smoothed_uniform.smoothed_uniform(inverse_smoothing_epsilon)

def MixtureUniform(
    mutual_information: float,
    dimensionality: int,
    normalize: bool,
    randomize_interactions: bool=False
) -> mixture.mixed_with_randomized_parameters:
    """
    Create a multivariate mixture of uniform distributions
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimensionality : int
        Dimensionality of the vectors.
    normalize : bool
        Normalize the distribution.
    randomize_interactions : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.

    Returns
    -------
    random_variable : mixture.mixture_uniform
        An instance of mixture.mixture_uniform
        with the provided value of the mutual information.
    """

    if randomize_interactions:
        raise NotImplementedError("Interaction randomization is not implemented for `MixtureUniform` yet.")

    componentwise_mutual_information = mutual_information / dimensionality
    
    return tools.stacked_multi_rv_frozen(mixture.mixed_with_randomized_parameters(componentwise_mutual_information, normalize), dimensionality)


def NoiselessChannel(
    mutual_information: float,
    permute: bool=False
) -> discrete.symmetric_noisy_channel:
    """
    Create a discrete noiseless channel with defined mutual information
    between the input and output.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    permute : bool, optional
        Apply random permutation.
        
    Returns
    -------
    random_variable : discrete.noiseless_channel
        An instance of discrete.noiseless_channel
        with the provided value of the mutual information.
    """

    probabilities = discrete.entropy_to_probabilities(mutual_information)
    alphabet = numpy.arange(len(probabilities))

    permutation = numpy.random.permutation(len(probabilities)) if permute else None
    
    return discrete.symmetric_noisy_channel(values=(alphabet, probabilities), permutation=permutation)


def SymmetricNoisyChannel(
    mutual_information: float,
    permute: bool=False,
    alphabet_size: int=None,
) -> discrete.symmetric_noisy_channel:
    """
    Create a discrete symmetric noisy channel with defined mutual information
    between the input and output.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    permute : bool, optional
        Apply random permutation.
    alphabet_size : int, optional
        Alphabet size. If `None`, selected as `ceil(exp(mutual_information))`.
        
    Returns
    -------
    random_variable : discrete.noiseless_channel
        An instance of discrete.noiseless_channel
        with the provided value of the mutual information.
    """

    if alphabet_size is None:
        alphabet_size = int(math.ceil(math.exp(mutual_information)))
        if alphabet_size < 2:
            alphabet_size = 2
        
    reroll_probability = discrete.mutual_information_to_reroll_probability(mutual_information, alphabet_size)
    probabilities = numpy.ones(alphabet_size) / alphabet_size
    alphabet = numpy.arange(alphabet_size)

    permutation = numpy.random.permutation(alphabet_size) if permute else None
    
    return discrete.symmetric_noisy_channel(values=(alphabet, probabilities), reroll_probability=reroll_probability, permutation=permutation)
