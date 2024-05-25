import numpy

from mutinfo.distributions.base import CorrelatedUniform


def test_uniform_distribution_statistics():
    """
    Test if `CorrelatedUniform` distribution possess the moments
    and other statistics of a uniform distribution.
    """

    n_samples = 1000
    atol = 1 / numpy.sqrt(n_samples)

    for X_dimension in range(1, 3):
        for Y_dimension in range(1, 3):
            for mutual_information in [0.0, 1.0, 5.0, 10.0]:
                random_variable = CorrelatedUniform(mutual_information, X_dimension, Y_dimension)

                x, y = random_variable.rvs(n_samples)
                x_y = numpy.concatenate([x, y], axis=1)

                x_y_mean = numpy.mean(x_y)
                assert abs(x_y_mean - 0.5) < atol, "A mean of the standard uniform distribution must be 1/2"

                x_y_var  = numpy.var(x_y)
                assert abs(x_y_var - 1.0 / 12.0) < atol, "A variance of the standard uniform distribution must be 1/12"

                x_y_min  = numpy.min(x_y)
                assert abs(x_y_min - 0.0) < atol, "The minimal value of a standard uniform distribution must be 0"

                x_y_max  = numpy.max(x_y)
                assert abs(x_y_max - 1.0) < atol, "The maximal value of a standard uniform distribution must be 1"