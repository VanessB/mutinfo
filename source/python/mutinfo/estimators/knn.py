import numpy
import math
from scipy.special import digamma, gamma
from sklearn.neighbors import BallTree, KDTree


_metric_tree_types = {
    "kd_tree": KDTree,
    "ball_tree": BallTree
}


class kNN_based:
    def __init__(self, k_neighbors: int=1, tree_type='kd_tree', tree_kwargs: dict={}):
        """
        Create a Kraskov-Stogbauer-Grassberger k-NN based
        mutual information estimator.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to use for estimation.
        tree_type : str, optional
            Specifies the type of metric tree used for estimation
            ('BallTree' or 'KDTree').
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

    def make_tree(self, data: numpy.array):
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
        
        return _metric_tree_types[self.tree_type](data, metric="chebyshev", **self.tree_kwargs)


class KSG(kNN_based):
    def __init__(self, k_neighbors: int=1, tree_type='kd_tree', tree_kwargs: dict={}):
        """
        Create a Kraskov-Stogbauer-Grassberger k-NN based
        mutual information estimator.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to use for estimation.
        tree_type : str, optional
            Specifies the type of metric tree used for estimation
            ('BallTree' or 'KDTree').
        tree_kwargs : dict, optional
            Metric tree additional arguments.

        References
        ----------
        .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
               information". Phys. Rev. E 69, 2004.
        """

        super().__init__(k_neighbors, tree_type, tree_kwargs)


    def __call__(self, x: numpy.array, y: numpy.array, std: bool=False) -> float:
        """
        Estimate the value of mutual information between random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x : array_like
            Samples from the first random vector.
        y : array_like
            Samples from the second random vector.
        std : bool
            Calculate standard deviation.

        Returns
        -------
        mutual information : float
            Estimated value of mutual information.
        """

        n_samples = x.shape[0]
        if y.shape[0] != n_samples:
            raise ValueError("The number of samples in `x` and `y` must be equal")

        k_neighbors = min(self.k_neighbors, n_samples-1)

        x = x.reshape(n_samples, -1)
        y = y.reshape(n_samples, -1)
        x_y = numpy.concatenate([x, y], axis=-1)

        X_dimension = x.shape[1]
        Y_dimension = y.shape[1]

        # Use metric trees for fast neares neighbors search.
        x_tree = self.make_tree(x)
        y_tree = self.make_tree(y)
        x_y_tree = self.make_tree(x_y)

        x_y_distances, _ = x_y_tree.query(x_y, k=(k_neighbors+1))
        x_y_distances = x_y_distances[:,k_neighbors] - 1.0e-15 # Add a small epsilon for tolerance.

        # Count marginal neighbors within x_y_distances.
        x_count = x_tree.query_radius(x, x_y_distances, count_only=True)
        y_count = y_tree.query_radius(y, x_y_distances, count_only=True)
        
        array = digamma(x_count) + digamma(y_count)
        mean = max(0.0, digamma(k_neighbors) + digamma(n_samples) - numpy.mean(array))

        if std:
            return mean, numpy.std(array) / math.sqrt(n_samples)
        else:
            return mean

        