import functools
import inspect
import math
import numpy

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, _is_fitted

from collections.abc import Callable
from typing import Any


class InformationEstimator(BaseEstimator):
    """
    Base class for estimators of information-theoretic quantities.
    """

    def __init__(self) -> None:
        pass

    def _check_arguments(self, arrays: tuple[numpy.ndarray]) -> None:
        """
        Check that given numpy arrays share the same length along their first axis.

        Parameters
        ----------
        arrays: tuple[numpy.ndarray]
            Arrays to check.

        Raises
        ------
        TypeError
            If any argument is not a numpy array.
        ValueError
            If an array is 0‑dimensional, or if the first‑axis lengths do not agree.
        """
        
        lengths: list[int] = []
        for array in arrays:
            if not isinstance(array, numpy.ndarray):
                raise TypeError(f"Expected each argument to be a numpy array, got {type(array).__name__}")
                
            if array.ndim == 0:
                raise ValueError(f"Expected each argument to be at least 1-dimensional; got 0 dimensions")
    
            lengths.append(array.shape[0])

        if len(set(lengths)) > 1:
            details = ", ".join(map(str, lengths))
            raise ValueError(
                f"Expected all arrays to have the same size along the first axis. "
                f"Got: {details}"
            )

    def check_arguments(function: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator to enforce that given array arguments share the same length
        along their first axis.
    
        Parameters
        ----------
        function : Callable[..., Any]
            Function to be wrapped.
    
        Returns
        -------
        wrapped : Callable[Callable[..., Any]
            A wrapped version of the original function with the shape check.
    
        Raises
        ------
        TypeError
            If any named argument is not a numpy array.
        ValueError
            If an array is 0‑dimensional, or if the first‑axis lengths do not agree.
        """

        def wrapped(self, *args, **kwargs) -> Any:
            self._check_arguments(args)

            return function(self, *args, **kwargs)

        return wrapped

    def check_arguments_named(*array_args_names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to enforce that given array arguments share the same length
        along their first axis (named variant).
    
        Parameters
        ----------
        *array_arg_names : str
            Names of the arguments (positional or keyword) that must be
            `array_like` instances with matching ``shape[0]``.
    
        Returns
        -------
        decorator : Callable[[Callable[..., Any]], Callable[..., Any]]
            A decorator that wraps the original function with the shape check.
    
        Raises
        ------
        TypeError
            If any named argument is not a numpy array.
        ValueError
            If an array is 0‑dimensional, or if the first‑axis lengths do not agree.
        """

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(function)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                signature = inspect.signature(function)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
    
                lengths: dict[str, int] = {}
                for name in array_args_names:
                    array = bound.arguments[name]
                    
                    if not isinstance(array, numpy.ndarray):
                        raise TypeError(f"Expected argument '{name}' to be a numpy array, got {type(array).__name__}")
                        
                    if array.ndim == 0:
                        raise ValueError(f"Expected argument '{name}' to be at least 1-dimensional; got 0 dimensions")

                    lengths[name] = array.shape[0]
    
                if len(set(lengths.values())) > 1:
                    details = ", ".join(f"{name}={length}" for name, length in lengths.items())
                    raise ValueError(
                        f"Expected all arrays to have the same size along the first axis. "
                        f"Got: {details}"
                    )

                return function(*args, **kwargs)
    
            return wrapper

        return decorator

    # Example usage of check_arguments_named.
    @check_arguments_named('x')
    def __call__(self, x: numpy.ndarray) -> float:
        """
        Estimate the value of the information-theoretic quantity using samples.

        Parameters
        ----------
        x : array_like
            Samples from random vectors.

        Returns
        -------
        estimate : float
            Estimated value of the quantity.
        """
        
        raise NotImplementedError


class TransformedInformationEstimator(InformationEstimator):
    """
    Base class for transform-based estimators
    (e.g., Sliced Mutual Information).

    The estimation pipeline is split into two clearly separated stages so that
    subclasses (and callers) can customize *aggregation* without having to
    re-implement the Monte-Carlo transform loop:

    * :method:`_sample_transformed_estimates` repeatedly applies ``transform`` to
      the inputs and evaluates ``estimator`` on each transformed pair, returning
      the raw per-transform estimates as a 1-D array. This is the geometry /
      "how to project" stage, fully delegated to ``transform``.
    * :method:`_aggregate` collapses that array into a single scalar. By default
      it is the arithmetic mean (recovering standard sliced MI). It can be
      overridden in a subclass, or replaced per-instance by passing an
      ``aggregate`` callable to the constructor.
    """

    _parameter_constraints: dict = {
        "estimator": [InformationEstimator],
        "transform": [TransformerMixin],
        "n_transform_samples": [int],
        "aggregate": [callable, None],
    }

    def __init__(
        self,
        estimator: InformationEstimator,
        transform: TransformerMixin,
        n_transform_samples: int=1,
        aggregate: Callable[[numpy.ndarray], float] | None=None,
    ) -> None:
        """
        Create an instance of `TransformedInformationEstimator` class.

        Parameters
        ----------
        estimator : InformationEstimator
            Backbone estimator.
        transform : TransformerMixin
            Transform to apply before the estimation.
        n_transform_samples : int, optional
            Non-negative number of Monte-Carlo samples,
            used in combination with random transforms.
        aggregate : Callable[[numpy.ndarray], float], optional
            Aggregation rule mapping the array of per-transform estimates to a
            single scalar. If ``None`` (default), the arithmetic mean is used,
            recovering the standard sliced-MI estimate. Subclasses may instead
            override :meth:`_aggregate`.
        """
        
        self.estimator = estimator
        self.transform = transform
        self.n_transform_samples = n_transform_samples
        self.aggregate = aggregate

        self._validate_params()

        if n_transform_samples < 1:
            raise ValueError("Expected `n_transform_samples` to be positive")

    def _sample_transformed_estimates(
        self,
        arrays: tuple[numpy.ndarray, ...],
    ) -> numpy.ndarray:
        """
        Apply the transform ``n_transform_samples`` times and evaluate the
        backbone estimator on each transformed tuple.

        Parameters
        ----------
        arrays : tuple[numpy.ndarray, ...]
            Samples from random vectors.

        Returns
        -------
        results : numpy.ndarray
            1-D array of per-transform estimates of length
            ``n_transform_samples``.
        """

        results = numpy.empty(self.n_transform_samples, dtype=numpy.float64)
        for transform_sample_index in range(self.n_transform_samples):
            transformed_arrays = self.transform.fit_transform(arrays)
            results[transform_sample_index] = self.estimator(*transformed_arrays)

        return results

    def _aggregate(self, values: numpy.ndarray) -> float:
        """
        Aggregate the per-transform estimates into a single scalar.

        Default behavior is the arithmetic mean (standard sliced MI). If an
        ``aggregate`` callable was supplied to the constructor, it is used
        instead. Subclasses requiring extra (introspectable) parameters --
        e.g. an inverse temperature -- should override this method directly.

        Parameters
        ----------
        values : numpy.ndarray
            Raw per-transform estimates.

        Returns
        -------
        estimate : float
            Aggregated estimate.
        """

        if self.aggregate is None:
            return float(values.mean())
        return float(self.aggregate(values))

    def __call__(
        self,
        *arrays : tuple[numpy.ndarray],
        std: bool=False
    ) -> float | tuple[float, float]:
        """
        Estimate the value of the information-theoretic quantity using samples
        from transformed random vectors.

        Parameters
        ----------
        arrays : tuple[array_like]
            Samples from random vectors.
        std : bool
            Also return the standard error of the raw per-transform estimates
            (computed *before* aggregation), as a diagnostic of transform
            variance.

        Returns
        -------
        estimate : float
            Aggregated estimate of the quantity.
        estimate_std : float
            Standard error of the raw per-transform estimates. Returned only
            if ``std=True``.
        """

        results = self._sample_transformed_estimates(arrays)
        estimate = self._aggregate(results)

        if std:
            return estimate, float(results.std() / math.sqrt(self.n_transform_samples))
        else:
            return estimate


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
            if not transform is None:
                transform.fit(x)

        return self

    def transform(self, X) -> tuple:
        check_is_fitted(self)

        return tuple(x if transform is None else transform.transform(x) for x, transform in zip(X, self.transforms))

    def __sklearn_is_fitted__(self) -> bool:
        return all(((transform is None) or _is_fitted(transform)) for transform in self.transforms)