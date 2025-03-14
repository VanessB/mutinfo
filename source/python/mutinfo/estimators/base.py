import numpy

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, _is_fitted

from collections.abc import Callable
from typing import Any


class MutualInformationEstimator(BaseEstimator):
    """
    Base class for mutual information estimators.
    """

    def __init__(self) -> None:
        pass

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

    def fit_transform(
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


class TransformedMutualInformationEstimator(MutualInformationEstimator):
    """
    Base class for transform-based mutual information estimators
    (e.g., Sliced Mutual Information).
    """

    _parameter_constraints: dict = {
        "estimator": [MutualInformationEstimator],
        "transform": [TransformerMixin],
        "n_transform_samples": [int],
    }

    def __init__(
        self,
        estimator: MutualInformationEstimator,
        transform: TransformerMixin,
        n_transform_samples: int=1,
    ) -> None:
        """
        Create an instance of `TransformedMutualInformationEstimator` class.

        Parameters
        ----------
        estimator : MutualInformationEstimator
            Backbone mutual information estimator.
        transform : TransformerMixin
            Transform to apply before the estimation.
        n_transform_samples : int, optional
            Non-negative number of Monte-Carlo samples,
            used in combination with random transforms.
        """
        
        self.estimator = estimator
        self.transform = transform
        self.n_transform_samples = n_transform_samples

        self._validate_params()

        #if not (n_transform_samples is None or isinstance(n_transform_samples, int)):
        #    raise TypeError("Expected `n_transform_samples` to be of type `int` or None.")
        if n_transform_samples < 1:
            raise ValueError("Expected `n_transform_samples` to be positive.")

    def __call__(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        std: bool=False
    ) -> float | tuple[float, float]:
        """
        Estimate the value of mutual information between two transformed
        random vectors using samples `x` and `y`.

        Parameters
        ----------
        x, y : array_like
            Samples from corresponding random vectors.
        std : bool
            Calculate standard deviation based on Monte-Carlo transform
            sampling.

        Returns
        -------
        estimate : float
            Estimated value.
        estimate_std : float or None
            Standard deviation of the estimate, or None if `std=False`
        """

        results = numpy.empty(self.n_transform_samples, dtype=numpy.float64)
        for transform_sample_index in range(self.n_transform_samples):
            transformed_x, transformed_y = self.transform.fit_transform((x, y))
            results[transform_sample_index] = self.estimator(transformed_x, transformed_y)

        if std:
            return results.mean(), results.std() / math.sqrt(self.n_projection_samples)
        else:
            return results.mean()


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