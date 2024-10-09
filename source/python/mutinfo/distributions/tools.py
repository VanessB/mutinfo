import numpy
from scipy.stats._multivariate import multi_rv_frozen


class stacked_multi_rv_frozen(multi_rv_frozen):
    def __init__(self, base_rv: multi_rv_frozen, dimensionality: int,
                 *args, **kwargs) -> None:
        """
        Create a multivariate random vector with a pushforward distribution
        of a random vactor `multi_rv_frozen` via a mapping `mapping`.

        Parameters
        ----------
        base_rv : scipy.stats._multivariate.multi_rv_frozen
            Base distribution.
        mapping : callable
            Transformation mapping.
        inverse_mapping : callable, optional
            Inverse of the transformation mapping.
        """

        super().__init__(*args, **kwargs)

        self._dist = base_rv
        self.dimensionality = dimensionality

    def rvs(self, size: int=1, *args, **kwargs) -> tuple[numpy.ndarray, numpy.ndarray]:
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
        
        return self._dist.rvs((size, self.dimensionality))

    @property
    def componentwise_mutual_information(self) -> numpy.ndarray:
        """
        Componentwise mutual information.

        Returns
        -------
        componentwise_mutual_information : np.array
            Componentwise mutual information.
        """
        return numpy.full(self.dimensionality, self._dist.mutual_information)

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        return numpy.sum(self.componentwise_mutual_information)
        

class mapped_multi_rv_frozen(multi_rv_frozen):
    def __init__(self, base_rv: multi_rv_frozen, mapping: callable,
                 inverse_mapping: callable=None, *args, **kwargs) -> None:
        """
        Create a multivariate random vector with a pushforward distribution
        of a random vactor `multi_rv_frozen` via a mapping `mapping`.

        Parameters
        ----------
        base_rv : scipy.stats._multivariate.multi_rv_frozen
            Base distribution.
        mapping : callable
            Transformation mapping.
        inverse_mapping : callable, optional
            Inverse of the transformation mapping.
        """

        super().__init__(*args, **kwargs)

        self._dist = base_rv
        self.mapping = mapping
        self.inverse_mapping = inverse_mapping

    def _check_inverse_mapping(function):
        def wrapper(self, *args, **kwargs):
            if self.inverse_mapping is None:
                raise NotImplementedError("Expected `inverse_mapping` to be defined")
            else:
                return function(self, *args, **kwargs)

        return wrapper
                
    @_check_inverse_mapping
    def _logcdf(self, x):
        return self._dist.logcdf(self.inverse_mapping(x))

    @_check_inverse_mapping
    def _cdf(self, x):
        return self._dist.cdf(self.inverse_mapping(x))

    def rvs(self, *args, **kwargs):
        return self.mapping(*self._dist.rvs(*args, **kwargs))

    @property
    def mutual_information(self) -> float:
        """
        Mutual information (under the ussumption that `self.mapping`
        is injective).

        Returns
        -------
        mutual_information : float
            Mutual information of the underlying distribution.
        """
        return self._dist.mutual_information