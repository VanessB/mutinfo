import numpy

from mutinfo.distributions.base import gamma_exponential


def test_mutual_information_and_inverse_shape_parameter():
    """
    Tests the functions which convert mutual information to
    inverse shape parameter and vice versa.
    """

    # `float` tests.
    assert gamma_exponential.inverse_shape_parameter_to_mutual_information(0.0) == 0.0, \
    "Zero inverse shape parameter must be equivalent to zero mutual information in the case of a gamma-exponential distribution."
    assert gamma_exponential.mutual_information_to_inverse_shape_parameter(0.0) == 0.0, \
    "Zero inverse shape parameter must be equivalent to zero mutual information in the case of a gamma-exponential distribution."

    true_mutual_information = 10.0
    assert abs(gamma_exponential.inverse_shape_parameter_to_mutual_information(
        gamma_exponential.mutual_information_to_inverse_shape_parameter(true_mutual_information)
    ) - true_mutual_information) < 1.0e-6, \
    "Mutual information to inverse shape parameter conversion is inconsistent (floats)."

    # NumPy tests.
    true_inverse_shape_parameter = numpy.array([[0.0, 1.0e-7, 1.0e-6, 1.0e-5], [5.0, 10.0, 200.0, 1.0e4]])
    true_mutual_information = numpy.array([[0.0, 0.5e-8, 0.5e-7, 0.5e-6], [1.320398016, 1.878830153, 4.729296455, 8.633289188]])
    
    inverse_shape_parameter = gamma_exponential.mutual_information_to_inverse_shape_parameter(true_mutual_information)
    print(inverse_shape_parameter - true_inverse_shape_parameter)
    assert numpy.allclose(true_inverse_shape_parameter, inverse_shape_parameter, atol=1.0e-5), "inverse shape parameter is calculated incorrectly."
    assert numpy.allclose(true_mutual_information, gamma_exponential.inverse_shape_parameter_to_mutual_information(inverse_shape_parameter)), \
    "Mutual information to inverse shape parameter conversion is inconsistent (NumPy arrays)."