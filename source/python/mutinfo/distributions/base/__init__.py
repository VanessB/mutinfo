import numpy
from scipy.special import ndtr, ndtri
from scipy.stats import ortho_group

from . import gamma_exponential
from . import normal
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
        


def _generate_cov_via_tridiagonal(mutual_information: float, X_dimension: int,
                                  Y_dimension: int, randomize_correlation: bool=True,
                                  randomize_interactions: bool=True) -> normal.CovViaTridiagonal:
    """
    Create a covariance matrix for a correlated multivariate normal distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimension of the first vector.
    Y_dimension : int
        Dimension of the first vector.
    randomize_correlation : bool, optional
        Randomize correlation coefficients (mutual information stays fixed).
        If not randomized, the correlation coefficients are equal and non-negative.
    randomize_interactions : bool, optional
        Use orthogonal matrices to randomize off-diagonal block of the
        covariation matrix (mutual information stays fixed).

    Returns
    -------
    random_variable : normal.CovViaTridiagonal
        An instance of normal.CovViaTridiagonal with defined mutual information.
    """

    min_dimension = min(X_dimension, Y_dimension)
    componentwise_mutual_information = _distribute_mutual_information(mutual_information, min_dimension, not randomize_correlation)
    correlation_coefficient = normal.mutual_information_to_correlation(componentwise_mutual_information)

    if randomize_interactions:
        X_orthogonal_matrix = None if X_dimension == 1 else ortho_group.rvs(X_dimension)
        Y_orthogonal_matrix = None if Y_dimension == 1 else ortho_group.rvs(Y_dimension)
    else:
        X_orthogonal_matrix = None
        Y_orthogonal_matrix = None

    return normal.CovViaTridiagonal(correlation_coefficient, X_orthogonal_matrix, Y_orthogonal_matrix)


def CorrelatedNormal(*args, **kwargs) -> normal.correlated_multivariate_normal:
    """
    Create a multivariate correlated normal distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimension of the first vector.
    Y_dimension : int
        Dimension of the first vector.
    randomize_correlation : bool, optional
        Randomize correlation coefficients (mutual information stays fixed).
        If not randomized, the correlation coefficients are equal and non-negative.
    randomize_interactions : bool, optional
        Use orthogonal matrices to randomize off-diagonal block of the
        covariation matrix (mutual information stays fixed).

    Returns
    -------
    random_variable : normal.correlated_multivariate_normal
        An instance of normal.correlated_multivariate_normal with
        defined mutual information.
    """

    covariance = _generate_cov_via_tridiagonal(*args, **kwargs)
    return normal.correlated_multivariate_normal(covariance)


def CorrelatedUniform(*args, **kwargs) -> mapped.mapped_multi_rv_frozen:
    """
    Create a multivariate correlated uniform distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimension of the first vector.
    Y_dimension : int
        Dimension of the first vector.
    randomize_correlation : bool, optional
        Randomize correlation coefficients (mutual information stays fixed).
        If not randomized, the correlation coefficients are equal and non-negative.
    randomize_interactions : bool, optional
        Use orthogonal matrices to randomize off-diagonal block of the
        covariation matrix (mutual information stays fixed).

    Returns
    -------
    random_variable : mapped.mapped_multi_rv_frozen
        An instance of mapped.mapped_multi_rv_frozen with
        defined mutual information and ndtr (normal to uniform) mapping.
    """

    # Use Gaussian CDF to acquire the uniform distribution.
    return mapped.mapped_multi_rv_frozen(CorrelatedNormal(*args, **kwargs), lambda x_y: (ndtr(x_y[0]), ndtr(x_y[1])), lambda x_y: (ndtri(x_y[0]), ndtri(x_y[1])))


def CorrelatedStudent(mutual_information: float, X_dimension: int,
                      Y_dimension: int, degrees_of_freedom: int,
                      randomize_correlation: bool=True,
                      randomize_interactions: bool=True) -> student.correlated_multivariate_student:
    """
    Create a multivariate correlated Student's distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    X_dimension : int
        Dimension of the first vector.
    Y_dimension : int
        Dimension of the first vector.
    degrees_of_freedom : int
        Number of degrees of freedom.
    randomize_correlation : bool, optional
        Randomize correlation coefficients (mutual information stays fixed).
        If not randomized, the correlation coefficients are equal and non-negative.
    randomize_interactions : bool, optional
        Use orthogonal matrices to randomize off-diagonal block of the
        covariation matrix (mutual information stays fixed).

    Returns
    -------
    random_variable : student.correlated_multivariate_student
        An instance of student.correlated_multivariate_student with
        defined mutual information and ndtr (normal to uniform) mapping.
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
        randomize_correlation,
        randomize_interactions,
    )
    return student.correlated_multivariate_student(covariance, degrees_of_freedom)


def SmoothedUniform(mutual_information: float,
                    X_dimension: int, Y_dimension: int,
                    randomize_smoothing_epsilon: bool=True) -> smoothed_uniform.smoothed_uniform:
    """
    Create a multivariate smoothed uniform distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimension : int
        Dimension of the first and the second vector.
    randomize_smoothing_epsilon : bool, optional
        Randomize smooting constants (mutual information stays fixed).
        If not randomized, the smooting epsilon constants are equal
        and non-negative.

    Returns
    -------
    random_variable : smoothed_uniform.smoothed_uniform
        An instance of smoothed_uniform.smoothed_uniform with
        defined mutual information.
    """

    min_dimension = min(X_dimension, Y_dimension)
    componentwise_mutual_information = _distribute_mutual_information(mutual_information, min_dimension, not randomize_smoothing_epsilon)
    inverse_smoothing_epsilon = smoothed_uniform.mutual_information_to_inverse_smoothing_epsilon(componentwise_mutual_information)
    
    return smoothed_uniform.smoothed_uniform(inverse_smoothing_epsilon, X_dimension, Y_dimension)


def GammaExponential(mutual_information: float,
                     X_dimension: int, Y_dimension: int,
                     randomize_shape_parameters: bool=True) -> gamma_exponential.gamma_exponential:
    """
    Create a multivariate gamma-exponential distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies within [0.0; +inf)).
    dimension : int
        Dimension of the first and the second vector.
    randomize_shape_parameters : bool, optional
        Randomize shape parameters (mutual information stays fixed).
        If not randomized, the shape parameters are equal
        and non-negative.

    Returns
    -------
    random_variable : gamma_exponential.gamma_exponential
        An instance of gamma_exponential.gamma_exponential with
        defined mutual information.
    """

    min_dimension = min(X_dimension, Y_dimension)
    componentwise_mutual_information = _distribute_mutual_information(mutual_information, min_dimension, not randomize_shape_parameters)
    inverse_shape_parameter = gamma_exponential.mutual_information_to_inverse_shape_parameter(componentwise_mutual_information)
    
    return gamma_exponential.gamma_exponential(inverse_shape_parameter, X_dimension, Y_dimension)