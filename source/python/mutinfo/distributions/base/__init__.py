import numpy

from scipy.special import ndtr, ndtri
from scipy.stats import ortho_group
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from . import gamma_exponential
from . import normal
from . import quantized
from . import smoothed_uniform
from . import student
from .. import mapped

from ...utils.checks import _check_dimension_value, _check_mutual_information_value


def _sample_from_simplex(dimension: int) -> numpy.ndarray:
    """
    Obtain a sample from the uniform distribution on a multidimensional simplex.

    Parameters
    ----------
    dimension : int
        Dimension of the simplex.

    Returns
    -------
    sample ; numpy.ndarray
        One sample from the uniform distribution on a multidimensional simplex.
    """
    
    result = numpy.random.exponential(scale=1.0, size=dimension)
    return result / numpy.sum(result)


def _distribute_mutual_information(mutual_information: float, dimension: int,
                                   uniform: bool=True) -> numpy.ndarray:
    """
    Uniformly or randomly distribute mutual information along dimensions.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimension : int
        Number of the dimensions.
    uniform : bool
        Distribute uniformly.

    Returns
    -------
    componentwise_mutual_information : numpy.ndarray
        1D array of componentwise mutual information values.
    """

    _check_dimension_value(dimension)

    if uniform:
        componentwise_mutual_information = mutual_information * numpy.ones(dimension) / dimension
    else:
        componentwise_mutual_information = mutual_information * _sample_from_simplex(dimension)

    return componentwise_mutual_information
        


def _generate_cov_via_tridiagonal(mutual_information: float,
                                  X_dimension: int, Y_dimension: int,
                                  randomize_interactions: bool=True,
                                  shuffle_interactions: bool=True) -> normal.CovViaTridiagonal:
    """
    Create a covariance matrix for a correlated multivariate normal distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimensionality of the first vector.
    Y_dimension : int
        Dimensionality of the first vector.
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

    min_dimension = min(X_dimension, Y_dimension)
    componentwise_mutual_information = _distribute_mutual_information(mutual_information, min_dimension, not randomize_interactions)
    correlation_coefficient = normal.mutual_information_to_correlation(componentwise_mutual_information)

    if shuffle_interactions:
        X_orthogonal_matrix = None if X_dimension == 1 else ortho_group.rvs(X_dimension)
        Y_orthogonal_matrix = None if Y_dimension == 1 else ortho_group.rvs(Y_dimension)
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
    X_dimension : int
        Dimensionality of the first vector.
    Y_dimension : int
        Dimensionality of the first vector.
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


def CorrelatedUniform(*args, **kwargs) -> mapped.mapped_multi_rv_frozen:
    """
    Create a multivariate correlated uniform distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimensionality of the first vector.
    Y_dimension : int
        Dimensionality of the first vector.
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
    return mapped.mapped_multi_rv_frozen(CorrelatedNormal(*args, **kwargs), lambda x, y: (ndtr(x), ndtr(y)), lambda x, y: (ndtri(x), ndtri(y)))


def CorrelatedStudent(mutual_information: float,
                      X_dimension: int, Y_dimension: int,
                      degrees_of_freedom: int,
                      randomize_interactions: bool=True,
                      shuffle_interactions: bool=True) -> student.correlated_multivariate_student:
    """
    Create a multivariate correlated Student's distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimensionality of the first vector.
    Y_dimension : int
        Dimensionality of the first vector.
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

    correction_term = student.mutual_information_correction_term(
        X_dimension, Y_dimension, degrees_of_freedom
    )

    correlation_mutual_information = mutual_information - correction_term
    if correlation_mutual_information < 0.0:
        raise ValueError(f"Impossible to create a correlated Student's distribution with the desired mutual information. The latter should be at least {correction_term:.2f} nat")

    covariance = _generate_cov_via_tridiagonal(
        correlation_mutual_information,
        X_dimension,
        Y_dimension,
        randomize_interactions,
        shuffle_interactions,
    )
    return student.correlated_multivariate_student(covariance, degrees_of_freedom)


def GammaExponential(mutual_information: float,
                     X_dimension: int, Y_dimension: int,
                     randomize_interactions: bool=True) -> gamma_exponential.gamma_exponential:
    """
    Create a multivariate gamma-exponential distribution
    given the value of the mutual information between the subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimensionality of the first vector.
    Y_dimension : int
        Dimensionality of the first vector.
    randomize_interactions : bool, optional
        Randomize component-wise mutual information
        (the total value of mutual information stays fixed).
        If not randomized, interactions are assigned uniformly.

    Returns
    -------
    random_variable : gamma_exponential.gamma_exponential
        An instance of gamma_exponential.gamma_exponential
        with the provided value of the mutual information.
    """

    min_dimension = min(X_dimension, Y_dimension)
    componentwise_mutual_information = _distribute_mutual_information(mutual_information, min_dimension, not randomize_interactions)
    inverse_shape_parameter = gamma_exponential.mutual_information_to_inverse_shape_parameter(componentwise_mutual_information)
    
    return gamma_exponential.gamma_exponential(inverse_shape_parameter, X_dimension, Y_dimension)


def UniformlyQuantized(mutual_information: float,
                       X_dimension: int, Y_dimension: int,
                       base_rv: rv_frozen,
                       randomize_interactions: bool=False) -> quantized.quantized_rv:
    """
    Create a two-dimensional mixed-type distribution
    given the value of the mutual information between the components.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimensionality of the first vector.
    Y_dimension : int
        Dimensionality of the first vector.
    base_rv : scipy.stats._multivariate.multi_rv_frozen
        Base univariate distribution for the first component.
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

    probabilities = quantized.entropy_to_probabilities(mutual_information / min(X_dimension, Y_dimension))
    quantiles = numpy.cumsum(probabilities)[:-1]

    return quantized.quantized_rv(base_rv, quantiles, X_dimension, Y_dimension)


def SmoothedUniform(mutual_information: float,
                    X_dimension: int, Y_dimension: int,
                    randomize_interactions: bool=True) -> smoothed_uniform.smoothed_uniform:
    """
    Create a multivariate smoothed uniform distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimensionality of the first vector.
    Y_dimension : int
        Dimensionality of the first vector.
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

    min_dimension = min(X_dimension, Y_dimension)
    componentwise_mutual_information = _distribute_mutual_information(mutual_information, min_dimension, not randomize_interactions)
    inverse_smoothing_epsilon = smoothed_uniform.mutual_information_to_inverse_smoothing_epsilon(componentwise_mutual_information)
    
    return smoothed_uniform.smoothed_uniform(inverse_smoothing_epsilon, X_dimension, Y_dimension)