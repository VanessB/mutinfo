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
    
    def __init__(
        self,
        estimator: MutualInformationEstimator,
        projection_dim: int=1,
        n_projection_samples: int=128,
    ) -> None:
        """
        Create a k-Sliced Mutual Information estimator

        Parameters
        ----------
        estimator : MutualInformationEstimator
            Base estimator used to estimate MI between projections.
        projection_dim : int or tuple of ints, optional
            Dimensionality of the projection subspace.
            Use a pair of ints to specify the dimensionalities for X and Y separately.
            If one of the argument should not be projected, use `None`.
        n_projection_samples : int, optional
            Number of Monte Carlo samples to estimate SMI.

        References
        ----------
        .. [1] Z. Goldfeld, K. Greenewald and T. Nuradha, "k-Sliced
               Mutual Information: A Quantitative Study of Scalability
               with Dimension". NeurIPS, 2022.
        """

        self.estimator = estimator
        self.n_projection_samples = n_projection_samples

        if isinstance(projection_dim, int):
            projection_dim = (projection_dim, projection_dim)
        elif isinstance(projection_dim, tuple):
            if len(projection_dim) != 2:
                raise ValueError("Expected `projection_dim` to have two elements.")
        else:
            raise ValueError("Expected `projection_dim` to be of type `int` or `tuple`.")
            
        self.projection_dim = projection_dim

    @staticmethod
    def generate_random_projection_matrix(dim: int, projection_dim: int) -> numpy.ndarray:
        """
        Sample a random projection matrix from the uniform distribution
        of orthogonal linear projectors from `dim` to `projection_dim`.

        Parameters
        ----------
        dim : int
            Dimensionality of the data to be projected.
        projection_dim : int
            Dimensionality after the projection.

        Returns
        -------
        Q : numpy.ndarray
            Orthogonal projection matrix
        """
        
        random_matrix = numpy.random.randn(dim, projection_dim)
        Q, _ = numpy.linalg.qr(random_matrix)

        return Q

    @staticmethod
    def project(data: numpy.ndarray, projection_dim: int) -> numpy.ndarray:
        """
        Project `data` orthogonally to a random linear subspace of
        dimensionality `projection_dim`.

        Parameters
        ----------
        data : numpy.ndarray
            Data to be projected.
        projection_dim : int
            Dimensionality after the projection.
            If `None`, no projection is applied.

        Returns
        -------
        Q : numpy.ndarray
            Orthogonal projection matrix
        """

        if projection_dim is None:
            return data

        Q = SMI.generate_random_projection_matrix(data.shape[1], projection_dim)
        return data @ Q

    def __call__(self, x: numpy.ndarray, y: numpy.ndarray, std: bool=False) -> float | tuple[float, float]:
        """
        Estimate the value of k-sliced mutual information between two random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x, y : array_like
            Samples from corresponding random vectors.
        std : bool
            Calculate standard deviation.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        mutual_information_std : float or None
            Standard deviation of the estimate, or None if `std=False`
        """

        results = numpy.empty(self.n_projection_samples, dtype=numpy.float64)
        for projection_sample_index in range(self.n_projection_samples):
            results[projection_sample_index] = self.estimator(
                self.project(x, self.projection_dim[0]),
                self.project(y, self.projection_dim[1])
            )

        if std:
            return results.mean(), results.std() / math.sqrt(self.n_projection_samples)
        else:
            return results.mean()