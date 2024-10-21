import numpy
import math
from scipy.special import digamma, gamma
from sklearn.neighbors import BallTree, KDTree

from .base import MutualInformationEstimator


_metric_tree_types = {
    "kd_tree": KDTree,
    "ball_tree": BallTree
}


class kNN_based(MutualInformationEstimator):
    def __init__(self, k_neighbors: int=1, tree_type: str='kd_tree', tree_kwargs: dict={}) -> None:
        """
        Create a k-NN based mutual information estimator.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to use for estimation.
        tree_type : {'ball_tree', 'kd_tree'}, optional
            Specifies the type of metric tree used for estimation.
            KDTree is used by defalt.
        tree_kwargs : dict, optional
            Metric tree additional arguments.
        """

        if k_neighbors < 1:
            raise ValueError("The number of neighbors must be at least 1")

        self.k_neighbors = k_neighbors

        if not tree_type in _metric_tree_types:
            raise ValueError(f"The `tree_type` must be in {set(_metric_tree_types.keys())}")

        self.tree_type = tree_type
        self.tree_kwargs = tree_kwargs

    def make_tree(self, data: numpy.ndarray) -> BallTree | KDTree:
        """
        Build a metric tree over the provided data.

        Parameters
        ----------
        data : array_like
            Data used to build the tree.

        Returns
        -------
        tree : BallTree or KDTree
            The metric tree over `data` points.
        """

        return _metric_tree_types[self.tree_type](data, **self.tree_kwargs)


class KSG(kNN_based):
    """
    Kraskov-Stogbauer-Grassberger k-NN based mutual information estimator.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """

    def __init__(self, k_neighbors: int=1, tree_type: str='kd_tree', tree_kwargs: dict={}) -> None:
        """
        Create a Kraskov-Stogbauer-Grassberger k-NN based
        mutual information estimator.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to use for estimation.
        tree_type : {'ball_tree', 'kd_tree'}, optional
            Specifies the type of metric tree used for estimation.
            KDTree is used by defalt.
        tree_kwargs : dict, optional
            Metric tree additional arguments.

        References
        ----------
        .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
               information". Phys. Rev. E 69, 2004.
        """

        tree_kwargs["metric"] = "chebyshev"
        super().__init__(k_neighbors, tree_type, tree_kwargs)


    def __call__(self, x: numpy.ndarray, y: numpy.ndarray, std: bool=False) -> float | tuple[float, float]:
        """
        Estimate the value of mutual information between two random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x, y : array_like
            Samples from corresponding random vectors.
        std : bool
            Calculate standard deviation.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        mutual_information_std : float or None
            Standard deviation of the estimate, or None if `std=False`
        """

        self._check_arguments(x, y)

        n_samples = x.shape[0]
        k_neighbors = min(self.k_neighbors, n_samples-1)

        x = x.reshape(n_samples, -1)
        y = y.reshape(n_samples, -1)
        x_y = numpy.concatenate([x, y], axis=-1)

        # Use metric trees for fast nearest neighbors search.
        x_tree = self.make_tree(x)
        y_tree = self.make_tree(y)
        x_y_tree = self.make_tree(x_y)

        x_y_distances, _ = x_y_tree.query(x_y, k=(k_neighbors+1))
        x_y_distances = x_y_distances[:,k_neighbors] - 1.0e-15 # Subtract a small epsilon for tolerance.

        # Count marginal neighbors within x_y_distances.
        x_count = x_tree.query_radius(x, x_y_distances, count_only=True)
        y_count = y_tree.query_radius(y, x_y_distances, count_only=True)

        array = digamma(x_count) + digamma(y_count)
        mean = max(0.0, digamma(k_neighbors) + digamma(n_samples) - numpy.mean(array))

        if std:
            return mean, numpy.std(array) / math.sqrt(n_samples)
        else:
            return mean
