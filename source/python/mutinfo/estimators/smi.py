import numpy
import math

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, _is_fitted

from .base import MutualInformationEstimator, TransformedMutualInformationEstimator


class slicing_based(BaseEstimator, TransformerMixin):
    """
    Base class for slicing-based transforms.
    """

    _parameter_constraints: dict = {
        "projection_dim": [int, list, tuple]
    }

    def __init__(
        self,
        projection_dim: int=1,
    ) -> None:
        """
        Create a slicing transform.

        Parameters
        ----------
        projection_dim : int or tuple of ints, optional
            Dimensionality of the projection subspace. Use a tuple of ints
            to specify the dimensionalities for different inputs separately.
            If one of the argument should not be projected, use `None`.
        """

        #if not (isinstance(projection_dim, int) or isinstance(projection_dim, tuple)):
        #    raise TypeError("Expected `projection_dim` to be of type `int` or `tuple`.")
            
        self.projection_dim = projection_dim
        self.projectors = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X) -> tuple:
        check_is_fitted(self)

        return tuple(x if projector is None else x @ projector for x, projector in zip(X, self.projectors))

    def __sklearn_is_fitted__(self) -> bool:
        return not (self.projectors is None)


class RandomSlicing(slicing_based):
    """
    Transform for the k-Sliced mutual information estimator.

    References
    ----------
    .. [1] Z. Goldfeld, K. Greenewald and T. Nuradha, "k-Sliced
           Mutual Information: A Quantitative Study of Scalability
           with Dimension". NeurIPS, 2022.
    """

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

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if not isinstance(X, tuple):
            raise ValueError("Expected `X` to be of type `tuple`")
        
        if isinstance(self.projection_dim, int):
            projection_dim = (self.projection_dim,) * len(X)
        elif len(X) != len(self.projection_dim):
            raise ValueError("Expected `X` and `self.projection_dim` to be of the same length")
        else:
            projection_dim = self.projection_dim
        
        self.projectors = [
            None if dim is None else RandomSlicing.generate_random_projection_matrix(x.shape[-1], dim)
            for x, dim in zip(X, projection_dim)
        ]

        return self

    
def SMI(
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
        Dimensionality of the projection subspace. Use a tuple of ints
        to specify the dimensionalities for different inputs separately.
        If one of the argument should not be projected, use `None`.
    n_projection_samples : int, optional
        Number of Monte-Carlo samples to estimate SMI.

    References
    ----------
    .. [1] Z. Goldfeld, K. Greenewald and T. Nuradha, "k-Sliced
           Mutual Information: A Quantitative Study of Scalability
           with Dimension". NeurIPS, 2022.
    """

    return TransformedMutualInformationEstimator(
        estimator=estimator,
        transform=RandomSlicing(projection_dim),
        n_transform_samples=n_projection_samples
    )