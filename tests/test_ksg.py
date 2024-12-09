import numpy

from mutinfo.distributions.base import CorrelatedNormal, CorrelatedUniform, CorrelatedStudent, SmoothedUniform, LogGammaExponential, UniformlyQuantized
from mutinfo.estimators.knn import KSG

from . import estimator_tester


def test_ksg_normal():
    """
    Test the KSG estimator on normal distributions.
    """

    estimator_tester.run_tests(
        KSG,
        CorrelatedNormal,
        numpy.linspace(0.0, 1.0, 5),
        "Bad KSG estimates for normal distribution",
        range(1, 3),
        range(1, 3),
        n_samples=10000,
        atol=0.05,
        rtol=0.05
    )

def test_ksg_uniform():
    """
    Test the KSG estimator on uniform distributions.
    """

    estimator_tester.run_tests(
        KSG,
        CorrelatedUniform,
        numpy.linspace(0.0, 1.0, 5),
        "Bad KSG estimates for uniform distribution",
        range(1, 3),
        range(1, 3),
        n_samples=10000,
        atol=0.05,
        rtol=0.05
    )

def test_ksg_student():
    """
    Test the KSG estimator on Student's distributions.
    """

    for degrees_of_freedom in range(1,3):
        estimator_tester.run_tests(
            KSG,
            lambda mutual_information, X_dimension, Y_dimension : CorrelatedStudent(mutual_information, X_dimension, Y_dimension, degrees_of_freedom),
            numpy.linspace(0.5, 1.0, 5),
            "Bad KSG estimates for Student's distribution",
            range(1, 3),
            range(1, 3),
            n_samples=10000,
            atol=0.1,
            rtol=0.05
        )

def test_ksg_smoothed_uniform():
    """
    Test the KSG estimator on smoothed uniform distributions.
    """

    estimator_tester.run_tests(
        KSG,
        lambda mutual_information, X_dimension, Y_dimension: SmoothedUniform(mutual_information, X_dimension),
        numpy.linspace(0.0, 1.0, 5),
        "Bad KSG estimates for smoothed uniform distribution",
        range(1, 3),
        n_samples=10000,
        atol=0.1,
        rtol=0.05
    )

def test_ksg_log_gamma_exponential():
    """
    Test the KSG estimator on log-gamma-exponential distributions.
    """

    estimator_tester.run_tests(
        KSG,
        lambda mutual_information, X_dimension, Y_dimension: LogGammaExponential(mutual_information, X_dimension),
        numpy.linspace(0.0, 1.0, 5),
        "Bad KSG estimates for log-gamma-exponential distribution",
        range(1, 3),
        n_samples=10000,
        atol=0.1,
        rtol=0.05
    )

def test_ksg_uniformly_quantized():
    """
    Test the KSG estimator on uniformly quantized distributions.
    """

    from scipy.stats import norm, uniform

    for distribution in [norm, uniform]:
        estimator_tester.run_tests(
            KSG,
            lambda mutual_information, X_dimension, Y_dimension: UniformlyQuantized(mutual_information, X_dimension, distribution(loc=0.0, scale=1.0)),
            numpy.linspace(0.0, 1.0, 5),
            f"Bad KSG estimates for uniformly quantized distribution ({distribution})",
            range(1, 3),
            n_samples=10000,
            atol=0.05,
            rtol=0.05
    )