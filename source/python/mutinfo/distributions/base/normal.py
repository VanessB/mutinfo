import numpy
from scipy.stats import Covariance, ortho_group
from scipy.stats._multivariate import multivariate_normal_frozen

from ...utils.checks import _check_dimension_value, _check_mutual_information_value


def _check_correlation_value(correlation_coefficient: float, name: str="correlation_coefficient") -> None:
    """
    Checks a correlation coefficient to be within (-1.0; 1.0)

    Parameters
    ----------
    correlation_coefficient : float or array_like
        Correlation coefficient (lies in (-1.0; 1.0)).
    name : str, optional
        Name of the variable to be checked.
        Default is "correlation_coefficient"
    """

    if numpy.any(correlation_coefficient <= -1.0) or numpy.any(correlation_coefficient >= 1.0):
        raise ValueError(f"Expected `{name}` to lie within (-1.0; 1.0)")

def mutual_information_to_correlation(mutual_information: float) -> float:
    """
    Calculate the absolute value of the correlation coefficient between two
    jointly Gaussian random variables given the value of mutual information.

    Parameters
    ----------
    mutual_information : float or array_like
        Mutual information (lies in [0.0; +inf)).

    Returns
    -------
    correlation_coefficient : float or array_like
        Corresponding correlation coefficient.
    """

    _check_mutual_information_value(mutual_information)

    return numpy.sqrt(1 - numpy.exp(-2.0 * mutual_information))

def correlation_to_mutual_information(correlation_coefficient: float) -> float:
    """
    Calculate the mutual information between two jointly Gaussian random
    variables given the correlation coefficient.

    Parameters
    ----------
    correlation_coefficient : float or array_like
        Correlation coefficient (lies in (-1.0; 1.0)).

    Returns
    -------
    mutual_information : float or array_like
        Corresponding mutual information.
    """

    _check_correlation_value(correlation_coefficient)

    return -0.5 * numpy.log(1.0 - correlation_coefficient**2)

def covariance_matrix_to_mutual_information(covariance_matrix: numpy.ndarray, split_dimension: int) -> float:
    """
    Calculate the mutual information between two jointly Gaussian random vectors
    given the covariance matrix.

    Parameters
    ----------
    covariance_matrix : np.array
        Symmetric positive semidefinite matrix.
    split_dimension : int
        Specifies the dimension of the first vector.

    Returns
    -------
    mutual_information : float
        Corresponding mutual information.
    """

    _check_dimension_value(split_dimension, "split_dimension")

    if split_dimension >= covariance_matrix.shape[0]:
        raise ValueError("Expected `split_dimension` to be less then covariance matrix dimension")

    try:
        _, X_Y_logabsdet = numpy.linalg.slogdet(covariance_matrix)
        _, X_logabsdet   = numpy.linalg.slogdet(covariance_matrix[:split_dimension,:split_dimension])
        _, Y_logabsdet   = numpy.linalg.slogdet(covariance_matrix[split_dimension:,split_dimension:])
    except ValueError as slogdet_error:
        raise ValueError("Covariance matrix must be symmetric and positive definite") from slogdet_error

    return 0.5 * (X_logabsdet + Y_logabsdet - X_Y_logabsdet)

def covariance_matrix_to_differential_entropy(covariance_matrix: numpy.ndarray) -> float:
    """
    Calculate the differential entropy of a multivariate Gaussian random vector
    given the covariance matrix.

    Parameters
    ----------
    covariance_matrix : np.array
        Symmetric positive semidefinite matrix.

    Returns
    -------
    differential_entropy : float
        Corresponding differential entropy.
    """

    try:
        dimension = covariance_matrix.shape[0]
        _, logabsdet = numpy.linalg.slogdet(covariance_matrix)
    except ValueError as slogdet_error:
        raise ValueError("Covariance matrix must be symmetric and positive definite") from slogdet_error

    return 0.5 * (dimension * math.log(2.0 * math.pi * math.e) + logabsdet)

def get_tridiagonal_whitening_parameters(correlation_coefficient: float) -> float:
    """
    Calculate the parameters (on- and off-diagonal elements)
    of the whitening transform.

    Parameters
    ----------
    correlation_coefficient : float or array_like
        Correlation coefficient (lies in (-1.0; 1.0)).

    Returns
    -------
    (on_diagoanl, off_diagonal) : tuple of float or array_like
        Corresponding whitening transformation matrix elements.
    """

    _check_correlation_value(correlation_coefficient)
    
    alpha = 0.5 / numpy.sqrt(1 + correlation_coefficient)
    beta  = 0.5 / numpy.sqrt(1 - correlation_coefficient)

    return (alpha + beta, alpha - beta)

def get_tridiagonal_colorizing_parameters(correlation_coefficient: float) -> float:
    """
    Calculate the parameters (on- and off-diagonal elements)
    of the colorizing transform.

    Parameters
    ----------
    correlation_coefficient : float or array_like
        Correlation coefficient (lies in (-1.0; 1.0)).

    Returns
    -------
    (on_diagoanl, off_diagonal) : tuple of float or array_like
        Corresponding colorizing transformation matrix elements.
    """

    _check_correlation_value(correlation_coefficient)
    
    alpha = 0.5 * numpy.sqrt(1 + correlation_coefficient)
    beta  = 0.5 * numpy.sqrt(1 - correlation_coefficient)

    return (alpha + beta, alpha - beta)


class CovViaTridiagonal(Covariance):
    def __init__(self, correlation_coefficient: numpy.ndarray,
                 X_orthogonal_matrix: numpy.ndarray=None,
                 Y_orthogonal_matrix: numpy.ndarray=None) -> None:
        """
        Create a Covariance object via a tridiagonal block form.
        This is a covariance matrix of jointly Gaussain random vectors
        with identity marginal covariation matrices.

        Parameters
        ----------
        correlation_coefficient : array_like
            1D array of correlation coefficients between random vectors.
        X_orthogonal_matrix : array_like
            Linear orthogonal mapping which is applied to the first vector.
        Y_orthogonal_matrix : array_like
            Linear orthogonal mapping which is applied to the second vector.
        """

        self._X_orthogonal_matrix = X_orthogonal_matrix
        self._Y_orthogonal_matrix = Y_orthogonal_matrix

        if len(correlation_coefficient.shape) != 1:
            raise ValueError("`correlation_coefficient` must be a 1D array")

        _check_correlation_value(correlation_coefficient)
        self._correlation_coefficient = correlation_coefficient
        
        min_dimension = correlation_coefficient.shape[0]
        self._X_dimension = min_dimension
        self._Y_dimension = min_dimension
        #self._max_dimension = max(self._X_dimension, self._Y_dimension)

        # Check matrices, avoid code repetition.
        for letter in ["X", "Y"]:
            matrix = getattr(self, f"_{letter}_orthogonal_matrix")
            if not matrix is None:
                try:
                    self._check_orthogonal_matrix(matrix)
                except ValueError as value_error:
                    raise ValueError(f"`{letter}_orthogonal_matrix` is not a square orthogonal matrix") from value_error

                setattr(self, f"_{letter}_dimension", matrix.shape[0])

        if (self._X_dimension != min_dimension and self._Y_dimension != min_dimension) or \
           (self._X_dimension < min_dimension or self._Y_dimension < min_dimension):
            raise ValueError("Dimensions of vectors can not be deduced; try checking the shapes of `correlation_coefficient` and `X/Y_orthogonal_matrix`")

        # Explicitly define covariance matrix.
        self._covariance = self._assemble_covariance_matrix()

        # Parameters for whitening and colorizing transformations.
        self._whitening_parameters  = get_tridiagonal_whitening_parameters(self._correlation_coefficient)
        self._colorizing_parameters = get_tridiagonal_colorizing_parameters(self._correlation_coefficient)

        self._allow_singular = False

    def _apply_tridiagonal(self, x: numpy.ndarray, y: numpy.ndarray,
                           on_diagonal: numpy.ndarray, off_diagonal: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Split the data and perform a tridiagonal transformation.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.
        y : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.
        on_diagonal : array_like
            Array of on-diagonal elements of the transform.
        off_diagonal : array_like
            Array of off-diagonal elements of the transform.
        

        Returns
        -------
        (X_, Y_) : tuple[numpy.ndarray]
            The transformed arrays of points.
        """

        for parameters in [on_diagonal, off_diagonal]:
            min_dimension = min(self._X_dimension, self._Y_dimension)
            if len(parameters.shape) != 1:
                raise ValueError(f"Expected `on_diagonal` and `off_diagonal` to be 1D arrays")
            if parameters.shape[0] != min_dimension:
                raise ValueError(f"Expected `on_diagonal` and `off_diagonal` to be arrays of length {min_dimension}")

        if self._X_dimension <= self._Y_dimension:
            return on_diagonal * x[...,:] + y[...,:self._X_dimension] * off_diagonal, \
                numpy.concatenate([        
                    on_diagonal * y[...,:self._X_dimension] + x[...,:] * off_diagonal,
                    y[...,self._X_dimension:]
                ], axis=-1)
        else:
            return numpy.concatenate([
                    on_diagonal * x[...,:self._Y_dimension] + y[...,:] * off_diagonal,
                    x[...,self._Y_dimension:],
                ], axis=-1), \
                on_diagonal * y[...,:] + x[...,:self._Y_dimension] * off_diagonal,

    def _whiten_colorize(self, x_y: numpy.ndarray, whiten: bool) -> numpy.ndarray:
        """
        Perform a whitening or colorizing transformation on data.

        Parameters
        ----------
        x_y : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.
        whiten : bool
            Specifies the transformation (True: whiten, False: colorize).

        Returns
        -------
        x_y_ : numpy.ndarray
            The transformed array of points.
        """

        if x_y.shape[-1] != self._X_dimension + self._Y_dimension:
            raise ValueError(
                f"The last dimension of `x` ({x.shape[-1]}) does not correspond with the dimensionality of the space" +
                f"({self._X_dimension} + {self._Y_dimension} = {self._X_dimension + self._Y_dimension})"
            )

        x = x_y[...,:self._X_dimension]
        y = x_y[...,self._X_dimension:]

        if whiten:
            if not self._X_orthogonal_matrix is None:
                x = numpy.dot(x, self._X_orthogonal_matrix)
    
            if not self._Y_orthogonal_matrix is None:
                y = numpy.dot(y, self._Y_orthogonal_matrix)
        
            on_diagonal, off_diagonal = self._whitening_parameters
        else:
            on_diagonal, off_diagonal = self._colorizing_parameters
            
        x, y = self._apply_tridiagonal(x, y, on_diagonal, off_diagonal)

        if not whiten:
            if not self._X_orthogonal_matrix is None:
                x = numpy.dot(x, self._X_orthogonal_matrix.T)
    
            if not self._Y_orthogonal_matrix is None:
                y = numpy.dot(y, self._Y_orthogonal_matrix.T)

        return numpy.concatenate([x, y], axis=-1)

    def _whiten(self, x_y: numpy.ndarray) -> numpy.ndarray:
        """
        Perform a whitening transformation on data.

        Parameters
        ----------
        x_y : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_y_ : numpy.ndarray
            The transformed array of points.
        """

        return self._whiten_colorize(x_y, whiten=True)
   
    def _colorize(self, x_y: numpy.ndarray) -> numpy.ndarray:
        """
        Perform a colorizing transformation on data.

        Parameters
        ----------
        x_y : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_y_ : numpy.ndarray
            The transformed array of points.
        """

        return self._whiten_colorize(x_y, whiten=False)    

    @property
    def componentwise_mutual_information(self) -> numpy.ndarray:
        """
        Componentwise mutual information.

        Returns
        -------
        componentwise_mutual_information : np.array
            Componentwise mutual information
        """
        return correlation_to_mutual_information(self._correlation_coefficient)

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information
        """
        return numpy.sum(self.componentwise_mutual_information)
    
    @property
    def _log_pdet(self) -> float:
        """
        Log of the pseudo-determinant of the covariance matrix.
        Equals to mutual information.
        """
        return self.mutual_information

    @property
    def _rank(self) -> int:
        """
        Rank of the covariance matrix.
        This type of matrices is always full-rank.
        """
        return self._X_dimension + self._Y_dimension

    @property
    def _shape(self) -> tuple[int, int]:
        """
        Shape of the covariance array
        """
        return (self._rank, self._rank)
    
    def _assemble_covariance_matrix(self) -> numpy.ndarray:
        """
        Assemble the covariance matrix.

        Returns
        -------
        covariance_matrix : nummpy.array
            Covariance matrix.
        """

        correlation_block = numpy.zeros((self._X_dimension, self._Y_dimension))
        numpy.fill_diagonal(correlation_block, self._correlation_coefficient)
        
        if not self._X_orthogonal_matrix is None:
            correlation_block = self._X_orthogonal_matrix @ correlation_block
        if not self._Y_orthogonal_matrix is None:
            correlation_block = correlation_block @ self._Y_orthogonal_matrix.T

        return numpy.block([
            [numpy.eye(self._X_dimension),      correlation_block      ],
            [     correlation_block.T,     numpy.eye(self._Y_dimension)]
        ])

    @staticmethod
    def _check_orthogonal_matrix(matrix: numpy.ndarray) -> None:
        """
        Checks a matrix for orthogonality and square shape.

        Parameters
        ----------
        matrix : array_like
            Matrix to check.

        Returns
        -------
        is_suitable : bool
            True if `matrix` is square and orthogonal.
        """

        if len(matrix.shape) != 2:
            raise ValueError("The matrix must be a 2D array")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The matrix must be square")

        if not numpy.allclose(matrix.T @ matrix, numpy.eye(matrix.shape[0])):
            raise ValueError("The matrix must be orthogonal")


class correlated_multivariate_normal(multivariate_normal_frozen):
    """
    Frozen multivariate normal distribution with known mutual information.
    """
    
    def __init__(self, cov: CovViaTridiagonal, **kwargs) -> None:
        """
        Create a frozen multivariate normal distribution with known mutual information.

        Parameters
        ----------
        mean : array_like, default: ``[0]``
            Mean of the distribution.
        cov : CovViaTridiagonal
            Tridiagonal symmetric positive (semi)definite covariance matrix of the
            distribution.
        """

        super().__init__(cov=cov, **kwargs)

    def rvs(self, *args, **kwargs) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        An adapter to a SciPy `multivariate_normal_frozen.rvs` to yield
        a pair of arrays instead of one.
        """

        x_y = super().rvs(*args, **kwargs)
        
        return x_y[...,:self.cov_object._X_dimension], x_y[...,self.cov_object._X_dimension:]

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """
        return self.cov_object.mutual_information