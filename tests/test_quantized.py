import math
import numpy

from scipy.special import xlogy
from scipy.stats import norm, uniform

from mutinfo.distributions.base import quantized, UniformlyQuantized


def test_entropy_and_quantiles():
    """
    Tests the functions which convert Shannon's entropy to
    quantiles and vice versa.
    """

    for n_labels in range(1, 100):
        true_entropy = max(0.0, math.log(n_labels) - 1.0e-16)
        probabilities = quantized.entropy_to_probabilities(true_entropy)
        assert numpy.allclose(numpy.full(n_labels, 1.0 / n_labels), probabilities), \
        "Distributions with log(integer) entropy should be uniform."

    for true_entropy in numpy.linspace(0.0, 10.0, 101):
        probabilities = quantized.entropy_to_probabilities(true_entropy)
        entropy = -numpy.sum(xlogy(probabilities, probabilities))
        assert abs(entropy - true_entropy) < 1.0e-6, \
        "Entropy to probabilities conversion is inconsistent."

def test_uniformly_quatized():
    """
    Run initialization tests for uniformly quantized random variable.
    """

    for base_rv in [norm(), uniform()]:
        for mutual_information in numpy.linspace(0.0, 10.0, 101):
            for dimensionality in range(1, 10):
                random_variable = UniformlyQuantized(mutual_information, dimensionality, base_rv)

                assert abs(mutual_information - random_variable.mutual_information) < 1.0e-9, \
                "The value of mutual information is inconsistent."

                for size in [1, 10, 100]:
                    x, y = random_variable.rvs(size)
                    assert x.shape == y.shape == (size, dimensionality), \
                    "The shape of a random sampling is incorrect."