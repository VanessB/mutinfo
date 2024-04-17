import numpy
from scipy.stats import ortho_group

from mutinfo.distributions.base import smoothed_uniform


def test_mutual_information_and_inverse_smoothing_epsilon():
    """
    Tests the functions which convert mutual information to
    inverse smoothing constant and vice versa.
    """

    # `float` tests.
    assert smoothed_uniform.inverse_smoothing_epsilon_to_mutual_information(0.0) == 0.0, \
    "Zero inverse smoothing epsilon must be equivalent to zero mutual information in the case of a smoothed uniform distribution."
    assert smoothed_uniform.mutual_information_to_inverse_smoothing_epsilon(0.0) == 0.0, \
    "Zero inverse smoothing epsilon must be equivalent to zero mutual information in the case of a smoothed uniform distribution."

    true_mutual_information = 10.0
    assert abs(smoothed_uniform.inverse_smoothing_epsilon_to_mutual_information(
        smoothed_uniform.mutual_information_to_inverse_smoothing_epsilon(true_mutual_information)
    ) - true_mutual_information) < 1.0e-6, \
    "Mutual information to inverse smoothing epsilon conversion is inconsistent (floats)."

    # NumPy tests.
    true_inverse_smoothing_epsilon = numpy.array([[0.0, 1.0, 2.0], [5.0, 10.0, 1.0e4]])
    true_mutual_information = numpy.array([[0.0, 0.25, 0.5], [1.116290732, 1.709437912, 8.517293191]])
    
    inverse_smoothing_epsilon = smoothed_uniform.mutual_information_to_inverse_smoothing_epsilon(true_mutual_information)
    assert numpy.allclose(true_inverse_smoothing_epsilon, inverse_smoothing_epsilon), "inverse smoothing epsilon is calculated incorrectly."
    assert numpy.allclose(true_mutual_information, smoothed_uniform.inverse_smoothing_epsilon_to_mutual_information(inverse_smoothing_epsilon)), \
    "Mutual information to inverse smooting epsilon conversion is inconsistent (NumPy arrays)."