import functools
import inspect
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
    """

    _parameter_constraints: dict = {
        "estimator": [InformationEstimator],
        "transform": [TransformerMixin],
        "n_transform_samples": [int],
    }

    def __init__(
        self,
        estimator: InformationEstimator,
        transform: TransformerMixin,
        n_transform_samples: int=1,
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
        """
        
        self.estimator = estimator
        self.transform = transform
        self.n_transform_samples = n_transform_samples

        self._validate_params()

        #if not (n_transform_samples is None or isinstance(n_transform_samples, int)):
        #    raise TypeError("Expected `n_transform_samples` to be of type `int` or None.")
        if n_transform_samples < 1:
            raise ValueError("Expected `n_transform_samples` to be positive")

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
            Calculate standard deviation based on Monte-Carlo transform
            sampling.

        Returns
        -------
        estimate : float
            Estimated value of the quantity.
        estimate_std : float or None
            Standard deviation of the estimate, or None if `std=False`
        """

        results = numpy.empty(self.n_transform_samples, dtype=numpy.float64)
        for transform_sample_index in range(self.n_transform_samples):
            transformed_arrays = self.transform.fit_transform(arrays)
            results[transform_sample_index] = self.estimator(*transformed_arrays)

        if std:
            return results.mean(), results.std() / math.sqrt(self.n_transform_samples)
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
            if not transform is None:
                transform.fit(x)

        return self

    def transform(self, X) -> tuple:
        check_is_fitted(self)

        return tuple(x if transform is None else transform.transform(x) for x, transform in zip(X, self.transforms))

    def __sklearn_is_fitted__(self) -> bool:
        return all(((transform is None) or _is_fitted(transform)) for transform in self.transforms)