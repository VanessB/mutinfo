import numpy

from mutinfo.distributions.base import CorrelatedNormal, CorrelatedUniform, CorrelatedStudent, SmoothedUniform, UniformlyQuantized
from mutinfo.estimators.knn import KSG

from . import estimator_tester


def test_ksg_normal():
    """
    Test the KSG estimator on normal distributions.
    """

    estimator_tester.run_tests(
        KSG,
        CorrelatedNormal,
        range(1, 3),
        range(1, 3),
        numpy.linspace(0.0, 1.0, 5),
        "Bad KSG estimates for normal distribution",
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
        range(1, 3),
        range(1, 3),
        numpy.linspace(0.0, 1.0, 5),
        "Bad KSG estimates for uniform distribution",
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
            range(1, 3),
            range(1, 3),
            numpy.linspace(0.5, 1.0, 5),
            "Bad KSG estimates for Student's distribution",
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
        SmoothedUniform,
        range(1, 3),
        range(1, 3),
        numpy.linspace(0.0, 1.0, 5),
        "Bad KSG estimates for smoothed uniform distribution",
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
            lambda mutual_information, X_dimension, Y_dimension: UniformlyQuantized(mutual_information, X_dimension, Y_dimension, distribution(loc=0.0, scale=1.0)),
            range(1, 3),
            range(1, 3),
            numpy.linspace(0.0, 1.0, 5),
            f"Bad KSG estimates for uniformly quantized distribution ({distribution})",
            n_samples=10000,
            atol=0.05,
            rtol=0.05
    )