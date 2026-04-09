import math
import numpy
from scipy.stats import ortho_group

from mutinfo.distributions.base import smoothed_uniform


def test_mutual_information_and_inverse_noise_scale():
    """
    Tests the functions which convert mutual information to inverse noise scale and vice versa.
    """

    # `float` tests.
    assert smoothed_uniform.inverse_noise_scale_to_mutual_information(0.0) == 0.0, \
    "Zero inverse noise scale must be equivalent to zero mutual information in the case of a smoothed uniform distribution."
    assert smoothed_uniform.mutual_information_to_inverse_noise_scale(0.0) == 0.0, \
    "Zero inverse noise scale must be equivalent to zero mutual information in the case of a smoothed uniform distribution."

    true_mutual_information = 10.0
    assert abs(smoothed_uniform.inverse_noise_scale_to_mutual_information(
        smoothed_uniform.mutual_information_to_inverse_noise_scale(true_mutual_information)
    ) - true_mutual_information) < 1.0e-6, \
    "Mutual information to inverse noise scale conversion is inconsistent (floats)."

    # NumPy tests.
    true_inverse_noise_scale = numpy.array([[0.0, 0.5, 1.0], [2.5, 5.0, 5.0e3]])
    true_mutual_information = numpy.array([[0.0, 0.25, 0.5], [1.116290732, 1.709437912, 8.517293191]])
    
    inverse_noise_scale = smoothed_uniform.mutual_information_to_inverse_noise_scale(true_mutual_information)
    assert numpy.allclose(true_inverse_noise_scale, inverse_noise_scale), "inverse noise scale is calculated incorrectly."
    assert numpy.allclose(true_mutual_information, smoothed_uniform.inverse_noise_scale_to_mutual_information(inverse_noise_scale)), \
    "Mutual information to inverse smooting epsilon conversion is inconsistent (NumPy arrays)."


def test_mutual_information_and_alphabet_size_and_inverse_noise_scale():
    """
    Tests the functions which convert mutual information to noise scale and vice versa.
    """

    # `float` tests.
    for alphabet_size in [2, 4, 8, 16, 32]:
        log_alphabet_size = math.log(alphabet_size)
        for inverse_noise_scale in [1.0, 2.0, 10.0, 100.0]:
            assert smoothed_uniform.inverse_noise_scale_and_alphabet_size_to_mutual_information(inverse_noise_scale, alphabet_size) == log_alphabet_size, \
            "Noise scale from [0.0; 1.0] must be equivalent to log(alphabet size) mutual information in the case of a smoothed discrete uniform distribution."
        assert smoothed_uniform.mutual_information_and_alphabet_size_to_inverse_noise_scale(log_alphabet_size, alphabet_size) == 1.0, \
        "Noise scale from [0.0; 1.0] must be equivalent to log(alphabet size) mutual information in the case of a smoothed discrete uniform distribution."

    true_mutual_information = 10.0
    alphabet_size = int(math.ceil(math.exp(true_mutual_information * 2.0)))
    assert abs(smoothed_uniform.inverse_noise_scale_and_alphabet_size_to_mutual_information(
        smoothed_uniform.mutual_information_and_alphabet_size_to_inverse_noise_scale(true_mutual_information, alphabet_size), alphabet_size
    ) - true_mutual_information) < 1.0e-6, \
    "Mutual information to noise scale conversion is inconsistent (floats)."

    # NumPy tests.
    # TODO!
    # true_inverse_noise_scale = numpy.array([[0.0, 0.5, 1.0], [2.5, 5.0, 5.0e3]])
    # true_mutual_information = numpy.array([[0.0, 0.25, 0.5], [1.116290732, 1.709437912, 8.517293191]])
    # 
    # inverse_noise_scale = smoothed_uniform.mutual_information_to_inverse_noise_scale(true_mutual_information)
    # assert numpy.allclose(true_inverse_noise_scale, inverse_noise_scale), "inverse noise scale is calculated incorrectly."
    # assert numpy.allclose(true_mutual_information, smoothed_uniform.inverse_noise_scale_to_mutual_information(inverse_noise_scale)), \
    # "Mutual information to inverse smooting epsilon conversion is inconsistent (NumPy arrays)."