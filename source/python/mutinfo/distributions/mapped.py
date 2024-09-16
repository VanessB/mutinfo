from scipy.stats._multivariate import multi_rv_frozen


class mapped_multi_rv_frozen(multi_rv_frozen):
    def __init__(self, base_rv: multi_rv_frozen, mapping: callable,
                 inverse_mapping: callable=None, *args, **kwargs) -> None:
        """
        Create a multivariate random vector with a pushforward distribution
        of a random vactor `multi_rv_frozen` via a mapping `mapping`.

        Parameters
        ----------
        base_rv : scipy.stats._multivariate.multi_rv_frozen
            The base random vector.
        mapping : callable
            The transformation mapping.
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
                raise NotImplementedError("`inverse_mapping` is required")
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