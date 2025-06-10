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

        X = self.uniform_randint.rvs((size, 1)) # (n_samples, dim)
        Y = X + self.uniform_delta.rvs((size, 1)) # (n_samples, dim)
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
        return 1 * (numpy.log(self.M) - (self.M - 1) * numpy.log(self.K) / self.M) # TODO: check.

class mixed_with_randomized_parameters(multi_rv_frozen):
    def __init__(self, mutual_information: float, normalize: bool, **kwargs) -> None:
        """
        Create a correlated variables from the mixture of dicrete and continuous
        distributions with known mutual information.

        Parameters
        ----------
        mutual_information : float
            Mutual information.
        normalize : bool
            Normalize the distribution.

        **kwargs ; dict
            Other
        """

        self._mutual_information = mutual_information
        self.uniform = uniform(0, 1)
        self.normalize = normalize
    
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

        lower_bound = numpy.ceil(numpy.exp(self.mutual_information))

        assert lower_bound >= 2, "lower_bound must be greater than or equal to 2"

        m = numpy.random.randint(lower_bound, 3*lower_bound, size)

        k = numpy.exp(m*(numpy.log(m)-self.mutual_information)/(m-1))

        x = numpy.floor(m*self.uniform.rvs(size))
        y = x + k * self.uniform.rvs(size)

        if self.normalize:
            x /= (3*lower_bound + k)
            y /= (3*lower_bound + k)
        
        return x, y

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        return self._mutual_information