import numpy
from scipy.stats import ortho_group

from . import normal


def _sample_from_simplex(dimension: int) -> numpy.array:
    """
    Uniformly sample a vector from the multidimensional simplex.

    Parameters
    ----------
    dimension : int
        Dimension of the simplex.

    Returns
    -------
    sample ; numpy.array
        A sample.
    """
    
    result = numpy.random.exponential(scale=1.0, size=dimension)
    return result / numpy.sum(result)


def CorrelatedNormal(mutual_information: float, X_dimension: int,
                     Y_dimension: int, randomize_correlation: bool=True,
                     randomize_interactions: bool=True) -> normal.correlated_multivariate_normal:
    """
    Generate correlated multivariate normal distribution
    with defined mutual information between subvectors.

    Parameters
    ----------
    mutual_information : float
        Mutual information (lies in [0.0; +inf)).
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
    random_variable : gaussian.correlated_multivariate_normal
        An instance of gaussian.correlated_multivariate_normal with
        defined mutual information.
    """

    min_dimension = min(X_dimension, Y_dimension)
    if min_dimension < 1:
        raise ValueError("Both dimensions must be greater then 0")

    if randomize_correlation:
        componentwise_mutual_information = mutual_information * _sample_from_simplex(min_dimension)
    else:
        componentwise_mutual_information = mutual_information * numpy.ones(min_dimension) / min_dimension

    correlation_coefficients = normal.mutual_information_to_correlation(componentwise_mutual_information)

    if randomize_interactions:
        X_orthogonal_matrix = None if X_dimension == 1 else ortho_group.rvs(X_dimension)
        Y_orthogonal_matrix = None if Y_dimension == 1 else ortho_group.rvs(Y_dimension)
    else:
        X_orthogonal_matrix = None
        Y_orthogonal_matrix = None

    covariance = normal.CovViaTridiagonal(correlation_coefficients, X_orthogonal_matrix, Y_orthogonal_matrix)
    
    return normal.correlated_multivariate_normal(covariance)