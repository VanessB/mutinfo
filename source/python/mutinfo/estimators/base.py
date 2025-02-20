import numpy

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, _is_fitted

from collections.abc import Callable
from typing import Any


class JointTransform(BaseEstimator, TransformerMixin):
    """
    Combination of transforms to be applied to elements of a tuple.
    """

    _parameter_constraints: dict = {
        "transforms": [list]
    }

    def __init__(self, transforms: list[TransformerMixin]) -> None:
        self.transforms = transforms

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        for x, transform in zip(X, self.transforms):
            transform.fit(x)

        return self

    def transform(self, X) -> tuple:
        check_is_fitted(self)

        return tuple(transform.transform(x) for x, transform in zip(X, self.transforms))

    def __sklearn_is_fitted__(self) -> bool:
        return all(_is_fitted(transform) for transform in self.transforms)


class MutualInformationEstimator(BaseEstimator):
    """
    Base class for mutual information estimators.
    """

    def __init__(self, preprocessor: TransformerMixin=None) -> None:
        self.preprocessor = preprocessor

    def _check_arguments(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        """
        Check samples `x` and `y`.

        Parameters
        ----------
        x : array_like
            Samples from the first random vector.
        y : array_like
            Samples from the second random vector.
        """

        if y.shape[0] != x.shape[0]:
            raise ValueError("The number of samples in `x` and `y` must be equal")

    def check_arguments(
        function: Callable[[numpy.ndarray, numpy.ndarray], Any]
    ) -> Callable[[numpy.ndarray, numpy.ndarray], Any]:
        """
        Wrapper for checking arguments.
        """

        def wrapped(self, x: numpy.ndarray, y: numpy.ndarray, *args, **kwargs) -> Any:
            self._check_arguments(x, y)
            return function(self, x, y, *args, **kwargs)

        return wrapped
        
    @check_arguments
    def __call__(self, x: numpy.ndarray, y: numpy.ndarray) -> float:
        """
        Estimate the value of mutual information between two random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x, y : array_like
            Samples from corresponding random vectors.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        """
        
        raise NotImplementedError