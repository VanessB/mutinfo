import numpy

from scipy.stats import uniform, randint
from scipy.stats._multivariate import multi_rv_frozen

class correlated_discrete_and_continuous_mixtures(multi_rv_frozen):
    def __init__(self, M: int, K: int, dim: int, **kwargs) -> None:
        """
        Create a correlated variables from the mixture of dicrete and continuous
        distributions with known mutual information.

        Parameters
        ----------
        M : int >= 2
            hyperparameter
        K : int >= 1
            hyperparameter
        dim : int
            distribution dimension

        **kwargs ; dict
            Other
        """

        self.M = M
        self.K = K
        self.dim = dim

        self.uniform_randint = randint(0, M)
        self.uniform_delta = uniform(0, K)

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

        X = self.uniform_randint.rvs((size, self.dim)) # (n_samples, dim)
        Y = X + self.uniform_delta.rvs((size, self.dim)) # (n_samples, dim)

        if self.dim != 1:
            return X, Y
        else:
            return X.flatten(), Y.flatten()

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        return self.dim * (numpy.log(self.M) - (self.M - 1) * numpy.log(self.K) / self.M) # TODO: check.