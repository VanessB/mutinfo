import numpy

from mutinfo.distributions.base import CorrelatedNormal, CorrelatedUniform, CorrelatedStudent, SmoothedUniform, LogGammaExponential, UniformlyQuantized
from mutinfo.estimators.knn import WKL

from . import estimator_tester


_k_neighbors_grid = [1, 5, 9]


def test_wkl_normal():
    """
    Test the WKL estimator on normal distributions.
    """

    for k_neighbors in _k_neighbors_grid:
        estimator_tester.run_tests(
            lambda : WKL(k_neighbors=k_neighbors),
            CorrelatedNormal,
            numpy.linspace(0.0, 1.0, 5),
            "Bad WKL estimates for normal distribution",
            range(1, 3),
            range(1, 3),
            n_samples=100000,
            atol=0.1,
            rtol=0.05
        )

def test_wkl_uniform():
    """
    Test the WKL estimator on uniform distributions.
    """

    for k_neighbors in _k_neighbors_grid:
        estimator_tester.run_tests(
            lambda : WKL(k_neighbors=k_neighbors),
            CorrelatedUniform,
            numpy.linspace(0.0, 1.0, 5),
            "Bad WKL estimates for uniform distribution",
            range(1, 3),
            range(1, 3),
            n_samples=100000,
            atol=0.1,
            rtol=0.05
        )

def test_wkl_student():
    """
    Test the WKL estimator on Student's distributions.
    """

    for degrees_of_freedom in range(1,3):
        for k_neighbors in _k_neighbors_grid:
            estimator_tester.run_tests(
                lambda : WKL(k_neighbors=k_neighbors),
                lambda mutual_information, X_dimension, Y_dimension : CorrelatedStudent(mutual_information, X_dimension, Y_dimension, degrees_of_freedom),
                numpy.linspace(0.5, 1.0, 5),
                "Bad WKL estimates for Student's distribution",
                range(1, 3),
                range(1, 3),
                n_samples=100000,
                atol=0.1,
                rtol=0.05
            )

def test_wkl_smoothed_uniform():
    """
    Test the WKL estimator on smoothed uniform distributions.
    """

    for k_neighbors in _k_neighbors_grid:
        estimator_tester.run_tests(
            lambda : WKL(k_neighbors=k_neighbors),
            lambda mutual_information, X_dimension, Y_dimension: SmoothedUniform(mutual_information, X_dimension),
            numpy.linspace(0.0, 1.0, 5),
            "Bad WKL estimates for smoothed uniform distribution",
            range(1, 3),
            n_samples=100000,
            atol=0.1,
            rtol=0.05
        )

def test_wkl_log_gamma_exponential():
    """
    Test the WKL estimator on log-gamma-exponential distributions.
    """

    for k_neighbors in _k_neighbors_grid:
        estimator_tester.run_tests(
            lambda : WKL(k_neighbors=k_neighbors),
            lambda mutual_information, X_dimension, Y_dimension: LogGammaExponential(mutual_information, X_dimension),
            numpy.linspace(0.2, 1.0, 5),
            "Bad WKL estimates for log-gamma-exponential distribution",
            range(1, 3),
            n_samples=100000,
            atol=0.1,
            rtol=0.05
        )