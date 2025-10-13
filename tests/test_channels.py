import math
import numpy

from mutinfo.distributions.base import discrete


def test_mutual_information_and_reroll_probability():
    """
    Tests the functions which convert mutual information to
    reroll probability and vice versa.
    """

    for alphabet_size in range(2, 10):
        # `float` tests.
        assert discrete.reroll_probability_to_mutual_information(1.0, alphabet_size) == 0.0, \
        "Reroll probability one must be equivalent to zero mutual information in the case of a symmetric noisy channel."
        assert discrete.mutual_information_to_reroll_probability(0.0, alphabet_size) == 1.0, \
        "Reroll probability one must be equivalent to zero mutual information in the case of a symmetric noisy channel."
    
        true_mutual_information = 0.5 * math.log(alphabet_size)
        assert abs(discrete.reroll_probability_to_mutual_information(
            discrete.mutual_information_to_reroll_probability(true_mutual_information, alphabet_size),
            alphabet_size
        ) - true_mutual_information) < 1.0e-6, \
        "Mutual information to reroll probability conversion is inconsistent (floats)."
    
        # NumPy tests.
        true_reroll_probability = numpy.array([[0.0, 1.0], [0.0, 1.0]])
        true_mutual_information = numpy.array([[math.log(alphabet_size), 0.0], [math.log(alphabet_size), 0.0]])
        
        reroll_probability = discrete.mutual_information_to_reroll_probability(true_mutual_information, alphabet_size)
        assert numpy.allclose(true_reroll_probability, reroll_probability, atol=1.0e-5), "Reroll probability is calculated incorrectly."
        assert numpy.allclose(true_mutual_information, discrete.reroll_probability_to_mutual_information(reroll_probability, alphabet_size)), \
        "Mutual information to reroll probability conversion is inconsistent (NumPy arrays)."