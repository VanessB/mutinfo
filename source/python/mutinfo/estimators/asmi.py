"""
Aligned Sliced Mutual Information (aSMI).

Unlike standard sliced mutual information, which draws an independent random
orthogonal projector for each input, aligned slicing uses a SHARED projection
matrix for `X` and `Y` when they share the same (flattened) ambient
dimensionality. This is the natural geometry when `X` and `Y` are different
representations of the same underlying object (e.g., a clean signal and a
noisy copy) living in a common ambient space.

Note that in this aligned geometry, vanishing aligned SMI does NOT in general
imply independence -- dependencies orthogonal to every shared direction are
invisible to the construction.

References
----------
.. [1] Free-Energy Sliced Mutual Information (Section 3, Aligned
       Free-Energy Slicing).
"""

import numpy

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted

from .base import InformationEstimator, TransformedInformationEstimator
from .smi import RandomOrthogonalProjector


class AlignedRandomSlicing(BaseEstimator, TransformerMixin):
    """
    Transform for the Aligned k-Sliced Mutual Information estimator.

    Uses a SHARED random orthogonal projection matrix for both inputs when
    their flattened ambient dimensionalities match; falls back to independent
    projections otherwise so the transform remains well-defined for arbitrary
    inputs.
    """

    _parameter_constraints: dict = {
        "projection_dim": [int]
    }

    def __init__(self, projection_dim: int=1) -> None:
        """
        Create an aligned random slicing transform.

        Parameters
        ----------
        projection_dim : int, optional
            Dimensionality of the projection subspace.
        """

        self.projection_dim = projection_dim

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("expected `X` to be a tuple of length 2 (x, y)")

        x_arr, y_arr = X
        x_flat = x_arr.reshape(x_arr.shape[0], -1)
        y_flat = y_arr.reshape(y_arr.shape[0], -1)

        dim_x = x_flat.shape[-1]
        dim_y = y_flat.shape[-1]

        proj_dim_x = min(self.projection_dim, dim_x)
        proj_dim_y = min(self.projection_dim, dim_y)

        if dim_x == dim_y:
            # Aligned case: share a single projection matrix.
            shared = RandomOrthogonalProjector.generate_random_projection_matrix(
                dim_x, proj_dim_x
            )
            self.projector_x_ = shared
            self.projector_y_ = shared
        else:
            # Fallback: independent projections (ambient dims differ).
            self.projector_x_ = RandomOrthogonalProjector.generate_random_projection_matrix(
                dim_x, proj_dim_x
            )
            self.projector_y_ = RandomOrthogonalProjector.generate_random_projection_matrix(
                dim_y, proj_dim_y
            )

        self.mean_x_ = (x_flat @ self.projector_x_).mean(axis=0)
        self.mean_y_ = (y_flat @ self.projector_y_).mean(axis=0)

        return self

    def transform(self, X) -> tuple:
        check_is_fitted(self)

        x_arr, y_arr = X
        x_flat = x_arr.reshape(x_arr.shape[0], -1)
        y_flat = y_arr.reshape(y_arr.shape[0], -1)

        return (
            x_flat @ self.projector_x_ - self.mean_x_,
            y_flat @ self.projector_y_ - self.mean_y_,
        )

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "projector_x_") and hasattr(self, "projector_y_")


def aSMI(
    estimator: InformationEstimator,
    projection_dim: int=1,
    n_projection_samples: int=128,
) -> TransformedInformationEstimator:
    """
    Create an Aligned k-Sliced Mutual Information estimator.

    Uses a SHARED random orthogonal projection matrix for `X` and `Y` when
    they share the same ambient dimensionality, and falls back to independent
    projections otherwise. Aggregates per-projection estimates via the
    standard arithmetic mean.

    Parameters
    ----------
    estimator : InformationEstimator
        Base estimator used to estimate MI between projections.
    projection_dim : int, optional
        Dimensionality of the projection subspace.
    n_projection_samples : int, optional
        Number of Monte-Carlo samples used to estimate aligned SMI.

    Returns
    -------
    estimator : TransformedInformationEstimator
        Wrapped estimator that, when called on `(x, y)`, returns aligned
        sliced mutual information.

    References
    ----------
    .. [1] Free-Energy Sliced Mutual Information (Section 3, Aligned
           Free-Energy Slicing).
    """

    return TransformedInformationEstimator(
        estimator=estimator,
        transform=AlignedRandomSlicing(projection_dim),
        n_transform_samples=n_projection_samples,
    )