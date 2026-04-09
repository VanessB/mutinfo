import numpy
from scipy import special


def log_hyperfactorial(n: numpy.typing.ArrayLike) -> float | numpy.ndarray:
    """
    Compute the natural logarithm of the hyperfactorial.

    The hyperfactorial of a non‑negative integer :math:`n` is defined as

    .. math::
        H(n) = \prod_{k=1}^{n} k^k.

    This function returns

    .. math::
        \log H(n) = \sum_{k=1}^{n} k \log k.

    Parameters
    ----------
    n : int or array_like of ints
        Input values, must be non‑negative integers.

    Returns
    -------
    float or ndarray
        The natural logarithm of the hyperfactorial for each element of `n`.
        Returns 0 for :math:`n = 0`. The output is a Python `float` if the
        input is a scalar, otherwise an `ndarray` of floats
    """

    n = numpy.asarray(n)
    is_scalar = numpy.isscalar(n)

    if not numpy.issubdtype(n.dtype, numpy.integer):
        raise TypeError("Input must be an integer or an array of integers")

    if numpy.any(n < 0):
        raise ValueError("Input values must be non‑negative")

    if n.size == 0:
        return numpy.zeros_like(n, dtype=float)

    max_n = n.max()
    if max_n == 0:
        result = numpy.zeros_like(n, dtype=float)
        return result.item() if is_scalar else result
    
    k = numpy.arange(0, max_n + 1)
    k_sum = numpy.cumsum(special.xlogy(k, k))
    result = k_sum[n]

    return result.item() if is_scalar else result