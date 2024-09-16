import numpy


class MutualInformationEstimator:
    """
    Base class for mutual information estimators.
    """
    
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