import numpy

from mutinfo.distributions.base import CorrelatedNormal
from mutinfo.estimators.ksg import KSG


def test_ksg_normal():
    """
    Test the KSG estimator on normal distributions
    """

    atol = 0.05
    rtol = 0.05
    n_samples = 10000

    estimator = KSG()
    for X_dimension in range(1,3):
        for Y_dimension in range(1,3):
            for mutual_information in numpy.linspace(0.0, 1.0, 5):
                random_variable = CorrelatedNormal(mutual_information, X_dimension, Y_dimension)
                x_y = random_variable.rvs(n_samples)
                estimated_mutual_information = estimator(x_y[:,X_dimension:], x_y[:,:X_dimension])

                assert numpy.allclose(mutual_information, estimated_mutual_information, atol=atol, rtol=rtol), \
                f"Bad KSG estimates for normal distribution ({X_dimension}, {Y_dimension}, {mutual_information})"