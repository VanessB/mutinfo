"""
Free-Energy Sliced Mutual Information (F-SMI / Gibbs-SMI).

Replaces the arithmetic mean over projections used in standard SMI by a
free-energy / log-sum-exp transform

    F_beta(Z) := (1 / beta) * log E[exp(beta * Z)],

where `Z` is the random projected mutual information. The inverse temperature
`beta` is a one-parameter knob:

- `beta = 0`: arithmetic mean (recovers standard sliced mutual information).
- `beta > 0`: optimistic / soft-max -- upweights highly informative projections.
- `beta < 0`: pessimistic / soft-min -- downweights highly informative projections.

The variational picture (Donsker-Varadhan) is that positive temperatures give
an entropy-regularized soft-maximization over projector laws, and negative
temperatures give an entropy-regularized soft-minimization.

This module exposes:
- `FreeEnergyTransformedInformationEstimator`: a thin subclass of
  `TransformedInformationEstimator` that only overrides the aggregation step,
  reusing the base Monte-Carlo transform loop.
- `FSMI`: free-energy version of standard (independent-projector) SMI.
- `FaSMI`: free-energy version of aligned SMI (shared projection matrix).

References
----------
.. [1] Free-Energy Sliced Mutual Information.
"""

import math

from scipy.special import logsumexp
from sklearn.base import TransformerMixin

from .base import InformationEstimator, TransformedInformationEstimator
from .smi import RandomSlicing
from .asmi import AlignedRandomSlicing


class FreeEnergyTransformedInformationEstimator(TransformedInformationEstimator):
    """
    Free-energy (Gibbs) aggregation variant of `TransformedInformationEstimator`.

    Reuses the base class' Monte-Carlo transform loop verbatim and only
    overrides :meth:`_aggregate`, replacing the arithmetic mean by the
    free-energy / log-sum-exp transform

        F_beta(Z) := (1 / beta) * log E[exp(beta * Z)].

    Limits:
    - `beta = 0`: arithmetic mean (recovers `TransformedInformationEstimator`).
    - `beta -> +inf`: maximum over samples (optimistic, max-type slicing).
    - `beta -> -inf`: minimum over samples (pessimistic, min-type slicing).
    """

    _parameter_constraints: dict = {
        "estimator": [InformationEstimator],
        "transform": [TransformerMixin],
        "n_transform_samples": [int],
        "beta": [float],
    }

    def __init__(
        self,
        estimator: InformationEstimator,
        transform: TransformerMixin,
        n_transform_samples: int=128,
        beta: float=0.0,
    ) -> None:
        """
        Create an instance of `FreeEnergyTransformedInformationEstimator`.

        Parameters
        ----------
        estimator : InformationEstimator
            Backbone estimator.
        transform : TransformerMixin
            Transform to apply before the estimation.
        n_transform_samples : int, optional
            Number of Monte-Carlo transform samples (must be >= 1).
        beta : float, optional
            Inverse temperature. Default `0.0` recovers the arithmetic mean.
        """

        # `beta` must be set *before* the parent constructor runs, because
        # `TransformedInformationEstimator.__init__` triggers parameter
        # validation, which reads every attribute named in this subclass'
        # `__init__` signature (including `beta`).
        self.beta = float(beta)

        super().__init__(
            estimator=estimator,
            transform=transform,
            n_transform_samples=n_transform_samples,
        )

    def _aggregate(self, values) -> float:
        """
        Free-energy / log-sum-exp aggregation of the per-projection estimates.

        Falls back to the arithmetic mean as the continuous `beta -> 0` limit.
        """

        if abs(self.beta) < 1.0e-12:
            # beta -> 0 limit: arithmetic mean (standard sliced estimate).
            return float(values.mean())

        # Numerically stable log-mean-exp.
        log_n = math.log(self.n_transform_samples)
        return float((logsumexp(self.beta * values) - log_n) / self.beta)


def FSMI(
    estimator: InformationEstimator,
    beta: float=0.0,
    projection_dim: int=1,
    n_projection_samples: int=128,
) -> FreeEnergyTransformedInformationEstimator:
    """
    Create a Free-Energy k-Sliced Mutual Information estimator (F-SMI / Gibbs-SMI).

    Uses the independent-projector geometry of standard SMI (i.e. separate
    random orthogonal projectors for `X` and `Y`), but replaces the
    arithmetic mean over projections by the free-energy transform.

    For the original `projection_dim = 1` geometry, F-SMI inherits the
    nullification property of SMI: it vanishes if and only if `X` and `Y`
    are independent.

    Parameters
    ----------
    estimator : InformationEstimator
        Base estimator used to estimate MI between projections.
    beta : float, optional
        Inverse temperature. Default `0.0` recovers standard SMI.
    projection_dim : int or tuple of ints, optional
        Dimensionality of the projection subspace.
    n_projection_samples : int, optional
        Number of Monte-Carlo samples used to estimate F-SMI.

    Returns
    -------
    estimator : FreeEnergyTransformedInformationEstimator
        Wrapped estimator that, when called on `(x, y)`, returns F-SMI.
    """

    return FreeEnergyTransformedInformationEstimator(
        estimator=estimator,
        transform=RandomSlicing(projection_dim),
        n_transform_samples=n_projection_samples,
        beta=beta,
    )


def FaSMI(
    estimator: InformationEstimator,
    beta: float=0.0,
    projection_dim: int=1,
    n_projection_samples: int=128,
) -> FreeEnergyTransformedInformationEstimator:
    """
    Create an Aligned Free-Energy k-Sliced Mutual Information estimator.

    Combines the aligned-slicing geometry (shared projection matrix between
    `X` and `Y` when ambient dimensions match) with the free-energy
    aggregation rule of `FSMI`.

    Parameters
    ----------
    estimator : InformationEstimator
        Base estimator used to estimate MI between projections.
    beta : float, optional
        Inverse temperature. Default `0.0` recovers aligned SMI.
    projection_dim : int, optional
        Dimensionality of the projection subspace.
    n_projection_samples : int, optional
        Number of Monte-Carlo samples used to estimate aligned F-SMI.

    Returns
    -------
    estimator : FreeEnergyTransformedInformationEstimator
        Wrapped estimator that, when called on `(x, y)`, returns aligned F-SMI.
    """

    return FreeEnergyTransformedInformationEstimator(
        estimator=estimator,
        transform=AlignedRandomSlicing(projection_dim),
        n_transform_samples=n_projection_samples,
        beta=beta,
    )