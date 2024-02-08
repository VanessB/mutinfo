import numpy
from scipy.stats import ortho_group

from mutinfo.distributions.base import normal


def test_mutual_information_and_correlation():
    """
    Tests the functions which convert mutual information to correlation
    coefficient and vice versa (Gaussian distribution case).
    """

    # `float` tests.
    assert normal.correlation_to_mutual_information(0.0) == 0.0, "Zero correlation coefficient must imply zero mutual information in the Gaussian case."
    assert normal.mutual_information_to_correlation(0.0) == 0.0, "Zero mutual information implies zero correlation coefficient."

    true_mutual_information = 10.0
    assert abs(normal.correlation_to_mutual_information(normal.mutual_information_to_correlation(true_mutual_information)) - true_mutual_information) < 1.0e-6, \
    "Mutual information to correlation coefficient conversion is inconsistent (floats)."

    # NumPy tests.
    true_mutual_information = numpy.array([[0.0, 1.0, 2.0], [10.0, 9.0, 8.0]])
    true_correlation_coefficient = numpy.array([[0.0, 0.929873495, 0.990799859], [0.999999999, 0.999999992, 0.999999944]])
    
    correlation_coefficient = normal.mutual_information_to_correlation(true_mutual_information)
    assert numpy.allclose(true_correlation_coefficient, correlation_coefficient), "Correlation coefficient is calculated incorrectly."
    assert numpy.allclose(true_mutual_information, normal.correlation_to_mutual_information(correlation_coefficient)), \
    "Mutual information to correlation coefficient conversion is inconsistent (NumPy arrays)."



def test_tridiagonal_covariance_whitening_and_colorization():
    """
    Test tridiagonal covariance matrix class and correspinding whitening and
    colorizing transforms.
    """

    # `float` tests.
    for correlation_coefficient in [-0.999, -0.9, -0.5, 0.0, 0.5, 0.9, 0.999]:
        colorizing_on_diagonal, colorizing_off_diagonal = normal.get_tridiagonal_colorizing_parameters(correlation_coefficient)
        whitening_on_diagonal,  whitening_off_diagonal  = normal.get_tridiagonal_whitening_parameters(correlation_coefficient)

        C = numpy.array([[colorizing_on_diagonal, colorizing_off_diagonal], [colorizing_off_diagonal, colorizing_on_diagonal]])
        W = numpy.array([[whitening_on_diagonal,  whitening_off_diagonal],  [whitening_off_diagonal,  whitening_on_diagonal ]])
        assert numpy.allclose(C @ W, numpy.eye(2)), "Colorization must be inverse of whitening"

    # NumPy tests.
    n_points = 19
    correlation_coefficient = numpy.linspace(-0.9, 0.9, n_points)
    colorizing_on_diagonal, colorizing_off_diagonal = normal.get_tridiagonal_colorizing_parameters(correlation_coefficient)
    whitening_on_diagonal,  whitening_off_diagonal  = normal.get_tridiagonal_whitening_parameters(correlation_coefficient)
    
    assert numpy.allclose(colorizing_on_diagonal * whitening_on_diagonal + colorizing_off_diagonal * whitening_off_diagonal,
                          numpy.ones(n_points)), "Colorizing must be the inverse of whitening"

    assert numpy.allclose(colorizing_off_diagonal * whitening_on_diagonal + colorizing_on_diagonal * whitening_off_diagonal,
                          numpy.zeros(n_points)), "Colorizing must be the inverse of whitening"

    # Whitening and colorizing tests.
    for X_dimension in range(1, 6):
        for Y_dimension in range(1, 6):
            min_dimension = min(X_dimension, Y_dimension)
            correlation_coefficients = numpy.linspace(-0.9, 0.9, min_dimension)

            X_orthogonal_matrix = ortho_group.rvs(X_dimension) if X_dimension > 1 else None
            Y_orthogonal_matrix = ortho_group.rvs(Y_dimension) if Y_dimension > 1 else None

            if X_dimension == Y_dimension:
                covariances = [
                    normal.CovViaTridiagonal(correlation_coefficients),
                    normal.CovViaTridiagonal(correlation_coefficients, X_orthogonal_matrix),
                    normal.CovViaTridiagonal(correlation_coefficients, Y_orthogonal_matrix=Y_orthogonal_matrix),
                    normal.CovViaTridiagonal(correlation_coefficients, X_orthogonal_matrix, Y_orthogonal_matrix),
                ]
            else:
                covariances = [normal.CovViaTridiagonal(correlation_coefficients, X_orthogonal_matrix, Y_orthogonal_matrix)]
            
            for covariance in covariances:
                for x in [
                    numpy.ones((X_dimension + Y_dimension)),
                    numpy.ones((10, X_dimension + Y_dimension)),
                    numpy.ones((20, 10, X_dimension + Y_dimension)),
                    numpy.ones((30, 20, 10, X_dimension + Y_dimension)),
                ]:
                    whitened_x  = covariance.whiten(x)
                    colorized_x = covariance.colorize(whitened_x)
                    assert numpy.allclose(x, colorized_x), "Colorizing must be the inverse of whitening"

    # Random colorizing tests.
    N_samples = 10000
    correlation_eps = 1.0e-11
    atol = 1.0e1 / numpy.sqrt(N_samples)
    for dimension in range(1, 8):
        correlation_coefficients = numpy.random.uniform(-1.0 + correlation_eps, 1.0 - correlation_eps, dimension)
        X_orthogonal_matrix, Y_orthogonal_matrix = (ortho_group.rvs(dimension), ortho_group.rvs(dimension)) if dimension > 1 else (None, None)
        covariance = normal.CovViaTridiagonal(correlation_coefficients, X_orthogonal_matrix, Y_orthogonal_matrix)
        
        x = numpy.random.normal(size=(N_samples, 2*dimension))
        colorized_x = covariance.colorize(x)
        empirical_cov = numpy.cov(colorized_x, rowvar=False)
        print(empirical_cov - covariance.covariance)
        assert numpy.allclose(empirical_cov, covariance.covariance, atol=atol)



def test_mutual_information_and_covariance():
    """
    Tests the function which calculates mutual information from a covariance
    matrix (Gaussian distribution case).
    """

    correlation_eps = 1.0e-11
    for X_dimension in range(1, 16):
        for Y_dimension in range(1, 16):
            X_Y_dimension = X_dimension + Y_dimension
        
            # Trivial test.
            assert normal.covariance_matrix_to_mutual_information(numpy.eye(X_Y_dimension), Y_dimension) == 0.0, \
            "Identity covariation matrix must imply zero mutual information in the Gaussian case."
        
            # Tests with random matrices.
            n_tests = 16
            for test in range(n_tests):
                correlation_coefficients = numpy.random.uniform(-1.0 + correlation_eps, 1.0 - correlation_eps, min(X_dimension, Y_dimension))
                covariance = normal.CovViaTridiagonal(correlation_coefficients, numpy.eye(X_dimension), numpy.eye(Y_dimension))
        
                assert abs(covariance.mutual_information - normal.covariance_matrix_to_mutual_information(covariance.covariance, X_dimension)) < 1.0e-10, \
                "Failed tests with random covariation matrices"