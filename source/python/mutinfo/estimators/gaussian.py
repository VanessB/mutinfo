import numpy
import math
from scipy.special import digamma, gamma, loggamma

from .base import InformationEstimator

class DifferentialEntropy(InformationEstimator):
    """Gaussian approximation of differential entropy.

    For a random vector X, computes h(X) = 0.5 * log((2 pi e)^d * det(Sigma)),
    where Sigma is the sample covariance matrix estimated with `bias=True`.
    """

    def __init__(self) -> None:
        super().__init__()

    @InformationEstimator.check_arguments_named('x')
    def __call__(self, x: numpy.ndarray) -> float:
        """
        Estimate differential entropy.

        Parameters
        ----------
        x : numpy.ndarray
            Data array of shape (n_samples, d) or (n_samples,).

        Returns
        -------
        float
            Gaussian differential entropy estimate.
        """

        x = x.reshape(x.shape[0], 1)
        d = x.shape[1]
        cov = numpy.cov(x, rowvar=False, bias=True)
        
        logdet = numpy.linalg.slogdet(cov)[1]
        
        return 0.5 * (d * (1.0 + math.log(2.0 * math.pi)) + logdet)


class TotalCorrelation(InformationEstimator):
    """
    Gaussian total correlation (multivariate mutual information).

    For k random vectors X_1, ..., X_k,
    TC(X_1, ..., X_k) = sum_i h(X_i) - h(X_1, ..., X_k).
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: numpy.ndarray) -> float:
        """
        Estimate total correlation.

        Parameters
        ----------
        *args : numpy.ndarray
            Variable number of arrays, each of shape (n_samples, d_i) or (n_samples,).

        Returns
        -------
        float
            Gaussian total correlation estimate.
        """

        arrays = [array.reshape(array.shape[0], 1) for array in args]

        joint = numpy.hstack(arrays)
        cov_joint = numpy.cov(joint, rowvar=False, bias=True)
        logdet_joint = numpy.linalg.slogdet(cov_joint)[1]

        sum_logdet = 0.0
        for array in arrays:
            if len(array.shape) == 1 or array.shape[1] == 1:
                sum_logdet += math.log(array.var(ddof=1))
            else:
                cov = numpy.cov(array, rowvar=False, bias=True)
                sum_logdet += numpy.linalg.slogdet(cov)[1]

        return 0.5 * (sum_logdet - logdet_joint)


class MutualInformation(TotalCorrelation):
    """
    Mutual information estimator for two random vectors.

    Alias for total correlation in the case of exactly two inputs.
    """

    def __call__(self, x: numpy.ndarray, y: numpy.ndarray) -> float:
        """Estimate mutual information between X and Y.

        Parameters
        ----------
        x : numpy.ndarray
            First random vector.
        y : numpy.ndarray
            Second random vector.

        Returns
        -------
        float
            Gaussian mutual information estimate.
        """
        return super().__call__(x, y)


class DualTotalCorrelation(InformationEstimator):
    """
    Gaussian dual total correlation (binding information).

    For k random vectors, DTC = sum_i h(X_{-i}) - (k-1) * h(X_1, ..., X_k),
    where X_{-i} denotes the joint vector of all variables except the i-th.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: numpy.ndarray) -> float:
        """Estimate dual total correlation.

        Parameters
        ----------
        *args : numpy.ndarray
            Variable number of arrays, each of shape (n_samples, d_i) or (n_samples,).

        Returns
        -------
        float
            Gaussian dual total correlation estimate.
        """

        n_arrays = len(args)
        if n_arrays == 1:
            return 0.0

        arrays = [array.reshape(array.shape[0], 1) for array in args]
        dims = [array.shape[1] for array in arrays]

        joint = numpy.hstack(arrays)
        cov_joint = numpy.cov(joint, rowvar=False, bias=True)
        logdet_joint = numpy.linalg.slogdet(cov_joint)[1]

        sum_logdet_leave_one = 0.0
        start = 0
        for index in range(n_arrays):
            end = start + dims[index]
            keep = list(range(start)) + list(range(end, cov_joint.shape[0]))
            
            cov_minus_i = cov_joint[numpy.ix_(keep, keep)]
            sum_logdet_leave_one += numpy.linalg.slogdet(cov_minus_i)[1]
            
            start = end

        return 0.5 * (sum_logdet_leave_one - (n_arrays - 1) * logdet_joint)