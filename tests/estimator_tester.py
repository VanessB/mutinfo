import numpy

def run_tests(estimator_factory, distribution_factory,
              MI_grid, message_on_fail,
              X_dimension_grid, Y_dimension_grid=None,
              n_samples=10000, atol=0.05, rtol=0.05):
    """
    Run tests on mutual information estimator.
    """

    for X_dimension in X_dimension_grid:
        for Y_dimension in ([X_dimension] if Y_dimension_grid is None else Y_dimension_grid):
            for mutual_information in MI_grid:
                estimator = estimator_factory()
                random_variable = distribution_factory(mutual_information, X_dimension, Y_dimension)

                x, y = random_variable.rvs(n_samples)
                estimated_mutual_information = estimator(x, y)

                assert numpy.allclose(mutual_information, estimated_mutual_information, atol=atol, rtol=rtol), \
                message_on_fail + f"({X_dimension}, {Y_dimension}, {mutual_information}, {estimated_mutual_information}, {estimator})"