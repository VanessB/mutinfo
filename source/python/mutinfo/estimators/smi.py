import numpy
import math

from .base import MutualInformationEstimator


class SMI(MutualInformationEstimator):
    """
    k-Sliced mutual information estimator.

    References
    ----------
    .. [1] Z. Goldfeld, K. Greenewald and T. Nuradha, "k-Sliced
           Mutual Information: A Quantitative Study of Scalability
           with Dimension". NeurIPS, 2022.
    """
    
    def __init__(self, estimator: callable, projection_dim: int=1,
                 n_projection_samples: int=128) -> None:
        """
        Create a k-Sliced Mutual Information estimator

        Parameters
        ----------
        estimator : callable
            Base estimator used to estimate MI between projections.
        projection_dim : int, optional
            Dimensionality of the projection subspace.
        n_projection_samples : int, optional
            Number of Monte Carlo samples to estimate SMI.

        References
        ----------
        .. [1] Z. Goldfeld, K. Greenewald and T. Nuradha, "k-Sliced
               Mutual Information: A Quantitative Study of Scalability
               with Dimension". NeurIPS, 2022.
        """

        self.estimator = estimator
        self.projection_dim = projection_dim
        self.n_projection_samples = n_projection_samples


    def generate_random_projection_matrix(self, dim: int) -> None:
        """
        Sample a random projection matrix from the uniform distribution
        of orthogonal linear projectors from `dim` to `self.projection_dim`

        Parameters
        ----------
        dim : int
            Dimension of the data to be projected

        Returns
        -------
        Q : np.array
            Orthogonal projection matrix
        """
        
        random_matrix = numpy.random.randn(dim, self.projection_dim)
        Q, _ = numpy.linalg.qr(random_matrix)

        return Q


    def __call__(self, x: numpy.ndarray, y: numpy.ndarray, std: bool=False) -> float:
        """
        Estimate the value of k-sliced mutual information between two random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x : array_like
            Samples from the first random vector.
        y : array_like
            Samples from the second random vector.
        std : bool
            Calculate standard deviation.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        mutual_information_std : float or None
            Standard deviation of the estimate, or None if `std=False`
        """

        self._check_arguments(x, y)

        results = numpy.empty(self.n_projection_samples, dtype=numpy.float64)
        for projection_sample_index in range(self.n_projection_samples):
            Q_X = self.generate_random_projection_matrix(x.shape[1])
            Q_Y = self.generate_random_projection_matrix(y.shape[1])
            
            results[projection_sample_index] = self.estimator(x @ Q_X, y @ Q_Y)

        return results.mean()