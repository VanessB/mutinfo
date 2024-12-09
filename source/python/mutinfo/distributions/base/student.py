import numpy
from scipy.special import digamma, loggamma
from scipy.stats import chi2
from scipy.stats._multivariate import multi_rv_frozen

from .normal import CovViaTridiagonal, correlated_multivariate_normal

from ...utils.checks import _check_dimension_value


def entropy_correction_term(dimensionality: int) -> float:
    """
    Calculate entropy correction term for Student's distribution.

    Parameters
    ----------
    dimensionality : int
        Dimensionality parameter.

    Returns
    -------
    correction_term : float
        Entropy correction term.
    """

    _check_dimension_value(dimensionality)

    half_dimension = 0.5 * dimensionality
    return loggamma(half_dimension) - half_dimension * digamma(half_dimension)


def mutual_information_correction_term(
    X_dim: int,
    Y_dim: int,
    degrees_of_freedom: int
) -> float:
    """
    Calculate entropy correction term for Student's distribution.
    
    Parameters
    ----------
    X_dim, Y_dim : int
        Dimensionality of corresponding vectors.
    degrees_of_freedom : int
        Number of dergees of freedom.

    Returns
    -------
    correction_term : float
        Entropy correction term.
    """

    _check_dimension_value(X_dim, "X_dim")
    _check_dimension_value(Y_dim, "Y_dim")

    return entropy_correction_term(degrees_of_freedom) + \
           entropy_correction_term(degrees_of_freedom + X_dim + Y_dim) - \
           entropy_correction_term(degrees_of_freedom + X_dim) - \
           entropy_correction_term(degrees_of_freedom + Y_dim)


class correlated_multivariate_student(multi_rv_frozen):
    def __init__(self, cov: CovViaTridiagonal, df: int, **kwargs) -> None:
        """
        Create a frozen multivariate Student's distribution with known mutual information.

        Parameters
        ----------
        cov : CovViaTridiagonal
            Tridiagonal symmetric positive (semi)definite covariance matrix of the
            auxiliary normal distribution.
        df : int
            Number of degrees of freedom.
        **kwargs ; dict
            Other 
        """

        self.normal = correlated_multivariate_normal(cov)
        self.chi2 = chi2(df=df, scale=1.0/df)

    def rvs(self, size: int=1) -> tuple[numpy.ndarray, numpy.ndarray]:
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
        
        normal_x, normal_y = self.normal.rvs(size)
        squared_magnitude = self.chi2.rvs(size)

        return normal_x * numpy.sqrt(squared_magnitude[:,None]), normal_y * numpy.sqrt(squared_magnitude[:,None])

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        return self.normal.mutual_information + \
               mutual_information_correction_term(
                   self.normal.cov_object._X_dim, self.normal.cov_object._Y_dim, self.degrees_of_freedom
               )