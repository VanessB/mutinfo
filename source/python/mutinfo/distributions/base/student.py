import numpy
from scipy.special import digamma, loggamma
from scipy.stats import chi2
from scipy.stats._multivariate import multi_rv_frozen

from .normal import CovViaTridiagonal, correlated_multivariate_normal


def entropy_correction_term(dimension: int) -> float:
    """
    Calculate entropy correction term for Student's distribution.

    Parameters
    ----------
    dimension : int
        Dimensionality parameter.

    Returns
    -------
    correction_term : float
        Entropy correction term.
    """

    half_dimension = 0.5 * dimension
    return loggamma(half_dimension) - half_dimension * digamma(half_dimension)


def mutual_information_correction_term(X_dimension: int, Y_dimension: int,
                                       degrees_of_freedom: int) -> float:
    """
    Calculate entropy correction term for Student's distribution.
    
    Parameters
    ----------
    X_dimension : int
        Dimension of the first vector.
    Y_dimension : int
        Dimension of the second vector.
    degrees_of_freedom : int
        Number of dergees of freedom.

    Returns
    -------
    correction_term : float
        Entropy correction term.
    """

    return entropy_correction_term(degrees_of_freedom) + \
           entropy_correction_term(degrees_of_freedom + X_dimension + Y_dimension) - \
           entropy_correction_term(degrees_of_freedom + X_dimension) - \
           entropy_correction_term(degrees_of_freedom + Y_dimension)


class correlated_multivariate_student(multi_rv_frozen):
    def __init__(self, cov: CovViaTridiagonal, df: int, **kwargs):
        """
        Create a frozen multivariate Student's distribution with known mutual information.

        Parameters
        ----------
        mean : array_like, default: ``[0]``
            Mean of the distribution.
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

    def rvs(self, size: int) -> numpy.array:
        """
        Random variate.

        Parameters
        ----------
        size : int
            Number of samples.

        Returns
        -------
        x_y : numpy.array
            Random sampling.
        """
        
        normal_x_y = self.normal.rvs(size)
        squared_magnitude = self.chi2.rvs(size)

        return normal_x_y * numpy.sqrt(squared_magnitude[:,None])

    @property
    def mutual_information(self):
        """
        Mutual information.
        """
        return self.normal.mutual_information + \
               mutual_information_correction_term(
                   self.normal.cov_object._X_dimension, self.normal.cov_object._Y_dimension, self.degrees_of_freedom
               )