import numpy


def _check_dimension_value(dimension: int, name: str="dimension") -> None:
    """
    Checks dimension to be integer and positive.

    Parameters
    ----------
    dimension : int
        Dimension, a positive integer.
    name : str, optional
        Name of the variable to be checked.
        Default is "dimension"
    """

    if not isinstance(dimension, int):
        raise TypeError(f"Expected `{name}` to be of type `int`")

    if dimension <= 0:
        raise ValueError(f"Expected `{name}` to be positive")


def _check_mutual_information_value(mutual_information: float, name: str="mutual_information") -> None:
    """
    Checks mutual information to be within [0.0; +inf)

    Parameters
    ----------
    mutual_information : float or array_like
        Mutual information (lies within [0.0; +inf)).
    name : str, optional
        Name of the variable to be checked.
        Default is "mutual_information"
    """

    if numpy.any(mutual_information < 0.0):
        raise ValueError(f"Expected `{name}` to be non-negative")


def _check_quantile_value(quantile: float, name: str="quantile") -> None:
    """
    Checks mutual information to be within [0.0; +inf)

    Parameters
    ----------
    quantile : float or array_like
        Quantile (lies within [0.0; 1.0]).
    name : str, optional
        Name of the variable to be checked.
        Default is "quantile"
    """

    if numpy.any(quantile < 0.0) or numpy.any(quantile > 1.0):
        raise ValueError(f"Expected `{name}` to be within (0;1)")