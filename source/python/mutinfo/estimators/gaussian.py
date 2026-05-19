import numpy
import math

from .base import InformationEstimator

class DifferentialEntropy(InformationEstimator):
    """
    Gaussian approximation of differential entropy.

    For a random vector X, computes h(X) = 0.5 * log((2 pi e)^d * det(Sigma)),
    where Sigma is the sample covariance matrix estimated with `bias=True`.
    """

    def __init__(self, biased: bool=False) -> None:
        """
        Initialize the Gaussian differential entropy estimator.

        Parameters
        ----------
        biased : bool, optional
            If True, use the biased covariance estimator (divide by n).
            If False (default), use the unbiased estimator (divide by n-1).
        """
        
        super().__init__()

        self.biased = biased

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
        differential_entropy: float
            Gaussian differential entropy estimate.
        """

        x = x.reshape(x.shape[0], -1)
        d = x.shape[1]
        cov = numpy.cov(x, rowvar=False, bias=self.biased)
        
        logdet = numpy.linalg.slogdet(cov)[1]
        
        return 0.5 * (d * (1.0 + math.log(2.0 * math.pi)) + logdet)


class TotalCorrelation(InformationEstimator):
    """
    Gaussian total correlation (multivariate mutual information).

    For k random vectors X_1, ..., X_k,
    TC(X_1, ..., X_k) = sum_i h(X_i) - h(X_1, ..., X_k).
    """

    def __init__(self, biased: bool=False) -> None:
        """
        Initialize the Gaussian total correlation estimator.

        Parameters
        ----------
        biased : bool, optional
            If True, use the biased covariance estimator (divide by n).
            If False (default), use the unbiased estimator (divide by n-1).
        """
        
        super().__init__()

        self.biased = biased

    @InformationEstimator.check_arguments
    def __call__(self, *arrays: numpy.ndarray) -> float:
        """
        Estimate total correlation.

        Parameters
        ----------
        *arrays : numpy.ndarray
            Variable number of arrays, each of shape (n_samples, d_i) or (n_samples,).

        Returns
        -------
        total_correlation: float
            Gaussian total correlation estimate.
        """

        is_one_array = len(arrays) == 1
        arrays = [array.reshape(array.shape[0], -1) for array in arrays]

        joint  = arrays[0] if is_one_array else numpy.hstack(arrays)
        cov    = numpy.cov(joint, rowvar=False, bias=self.biased)
        logdet = numpy.linalg.slogdet(cov)[1]

        # Treat one array as a stack of d one-dimensional arrays.
        if is_one_array:
            sum_logdet_diag = numpy.log(numpy.diag(cov)).sum()
        else:
            sum_logdet_diag = 0.0
            start = 0
            for array in arrays:
                size = array.shape[-1]
                end  = start + size
                
                block = cov[start:end,start:end]
                sum_logdet_diag += numpy.linalg.slogdet(block)[1]

                start = end
    
        return 0.5 * (sum_logdet_diag - logdet)


class MutualInformation(TotalCorrelation):
    """
    Mutual information estimator for two random vectors.

    Alias for total correlation in the case of exactly two inputs.
    """

    @InformationEstimator.check_arguments_named('x', 'y')
    def __call__(self, x: numpy.ndarray, y: numpy.ndarray) -> float:
        """
        Estimate mutual information between X and Y.

        Parameters
        ----------
        x : numpy.ndarray
            First random vector.
        y : numpy.ndarray
            Second random vector.

        Returns
        -------
        mutual_information: float
            Gaussian mutual information estimate.
        """

        return super().__call__(x, y)


class DualTotalCorrelation(InformationEstimator):
    """
    Gaussian dual total correlation (binding information).

    For k random vectors, DTC = sum_i h(X_{-i}) - (k-1) * h(X_1, ..., X_k),
    where X_{-i} denotes the joint vector of all variables except the i-th.
    """

    def __init__(self, biased: bool=False) -> None:
        """
        Initialize the Gaussian dual total correlation estimator.

        Parameters
        ----------
        biased : bool, optional
            If True, use the biased covariance estimator (divide by n).
            If False (default), use the unbiased estimator (divide by n-1).
        """
        
        super().__init__()

        self.biased = biased

    @InformationEstimator.check_arguments
    def __call__(self, *arrays: numpy.ndarray) -> float:
        """
        Estimate dual total correlation.

        Parameters
        ----------
        *arrays : numpy.ndarray
            Variable number of arrays, each of shape (n_samples, d_i) or (n_samples,).

        Returns
        -------
        dual_total_correlation: float
            Gaussian dual total correlation estimate.
        """

        n_arrays = len(arrays)
        is_one_array = n_arrays == 1
        arrays = [array.reshape(array.shape[0], -1) for array in arrays]

        joint  = arrays[0] if is_one_array else numpy.hstack(arrays)
        cov    = numpy.cov(joint, rowvar=False, bias=self.biased)
        logdet = numpy.linalg.slogdet(cov)[1]

        # Treat one array as a stack of d one-dimensional arrays.
        if is_one_array:
            # Efficient calculation using the inverse matrix trick.
            inverse_cov = numpy.linalg.inv(cov)
            return 0.5 * (numpy.log(numpy.diag(inverse_cov)).sum() + logdet)
        else:
            sum_logdet_leave_one = 0.0
            start = 0
            for array in arrays:
                size = array.shape[-1]
                end  = start + size
                
                keep = list(range(start)) + list(range(start + size, cov.shape[0]))
                block = cov[numpy.ix_(keep, keep)]
                sum_logdet_leave_one += numpy.linalg.slogdet(block)[1]
                
                start = end
    
            return 0.5 * (sum_logdet_leave_one - (n_arrays - 1) * logdet)