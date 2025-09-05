import numpy

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, _is_fitted

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from .base import MutualInformationEstimator, TransformedMutualInformationEstimator, JointTransform


class flattening_transform(BaseEstimator, TransformerMixin):
    """
    Transform for flattening tensors.
    """

    def __init__(self) -> None:
        pass

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> numpy.ndarray:
        check_is_fitted(self)

        return X.reshape(X.shape[0], -1)

    def __sklearn_is_fitted__(self) -> bool:
        return True


class slicing_based(JointTransform):
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

        super().__init__([None])

        #if not (isinstance(projection_dim, int) or isinstance(projection_dim, tuple)):
        #    raise TypeError("Expected `projection_dim` to be of type `int` or `tuple`.")
            
        self.projection_dim = projection_dim

    def fit_projetion_dim(self, X) -> tuple[int]:
        if not isinstance(X, tuple):
            raise ValueError("Expected `X` to be of type `tuple`")
        
        if isinstance(self.projection_dim, int):
            projection_dim = (self.projection_dim,) * len(X)
        elif len(X) != len(self.projection_dim):
            raise ValueError("Expected `X` and `self.projection_dim` to be of the same length")
        else:
            projection_dim = self.projection_dim

        return projection_dim


class RandomOrthogonalProjector(BaseEstimator, TransformerMixin):
    """
    Random orthogonal projector.
    """

    _parameter_constraints: dict = {
        "projection_dim": [int]
    }

    def __init__(self, projection_dim: int) -> None:
        self.projection_dim = projection_dim
        self.mean = None
        self.projector = None

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
        self.projector = RandomOrthogonalProjector.generate_random_projection_matrix(X.shape[-1], self.projection_dim)
        self.mean = (X @ self.projector).mean(axis=0)

        return self

    def transform(self, X) -> numpy.ndarray:
        check_is_fitted(self)

        return X @ self.projector - self.mean

    def __sklearn_is_fitted__(self) -> bool:
        return (not self.projector is None) and (not self.mean is None)


class RandomSlicing(slicing_based):
    """
    Transform for the k-Sliced Mutual Information estimator.

    References
    ----------
    .. [1] Z. Goldfeld, K. Greenewald and T. Nuradha, "k-Sliced
           Mutual Information: A Quantitative Study of Scalability
           with Dimension". NeurIPS, 2022.
    """

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        projection_dim = self.fit_projetion_dim(X)
        
        self.transforms = [
            None if dim is None else Pipeline([
                ("flattening", flattening_transform()),
                ("projection", RandomOrthogonalProjector(min(dim, numpy.prod(x.shape[1:]))))
            ])
            for x, dim in zip(X, projection_dim)
        ]

        return super().fit(X, y)


class PrincipleComponentSlicing(slicing_based):
    """
    Transform for the Principle Component Sliced Mutual Information estimator.

    References
    ----------
    .. [1] TODO
    """

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        projection_dim = self.fit_projetion_dim(X)
        
        self.transforms = [
            None if dim is None else Pipeline([
                ("flattening", flattening_transform()),
                ("projection", PCA(n_components=min(dim, numpy.prod(x.shape[1:])), whiten=True))
            ])
            for x, dim in zip(X, projection_dim)
        ]

        return super().fit(X, y)


class CanonicalCorrelationSlicing(BaseEstimator, TransformerMixin):
    """
    Transform for the Canonical Correlation Sliced Mutual Information estimator.

    References
    ----------
    .. [1] TODO
    """

    _parameter_constraints: dict = {
        "projection_dim": [int]
    }

    def __init__(self, projection_dim: int=1) -> None:
        self.projection_dim = projection_dim
        self.cca = None
        self.flattening_x = None
        self.flattening_y = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        projection_dim = min(self.projection_dim, numpy.prod(X[0].shape[1:]), numpy.prod(X[1].shape[1:]))
        self.cca = CCA(projection_dim).fit(*X)
        self.flattening_x = flattening_transform().fit(X[0])
        self.flattening_y = flattening_transform().fit(X[1])
        
        return self

    def transform(self, X, y=None):
        return self.cca.transform((self.flattening_x(X[0]), self.flattening_x(X[1])))

    def __sklearn_is_fitted__(self) -> bool:
        return (not self.cca is None) and _is_fitted(self.cca)

    
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


def PCMI(
    estimator: MutualInformationEstimator,
    projection_dim: int=1,
) -> None:
    """
    Create a Principle Component Mutual Information estimator

    Parameters
    ----------
    estimator : MutualInformationEstimator
        Base estimator used to estimate MI between projections.
    projection_dim : int or tuple of ints, optional
        Dimensionality of the projection subspace. Use a tuple of ints
        to specify the dimensionalities for different inputs separately.
        If one of the argument should not be projected, use `None`.

    References
    ----------
    .. [1] TODO
    """

    return TransformedMutualInformationEstimator(
        estimator=estimator,
        transform=PrincipleComponentSlicing(projection_dim)
    )


def CCMI(
    estimator: MutualInformationEstimator,
    projection_dim: int=1,
) -> None:
    """
    Create a Canonical Correlation Mutual Information estimator

    Parameters
    ----------
    estimator : MutualInformationEstimator
        Base estimator used to estimate MI between projections.
    projection_dim : int, optional
        Dimensionality of the projection subspace. Use a tuple of ints
        to specify the dimensionalities for different inputs separately.
        If one of the argument should not be projected, use `None`.

    References
    ----------
    .. [1] TODO
    """

    return TransformedMutualInformationEstimator(
        estimator=estimator,
        transform=CanonicalCorrelationSlicing(projection_dim)
    )