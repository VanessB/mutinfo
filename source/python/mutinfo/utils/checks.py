import numpy


def _check_mutual_information_value(mutual_information: float):
    """
    Checks mutual information to be within [0.0; +inf)

    Parameters
    ----------
    mutual_information : float or array_like
        Mutual information (lies within [0.0; +inf)).
    """

    if numpy.any(mutual_information < 0.0):
        raise ValueError("Mutual information must be non-negative")