import numpy

from mutinfo.distributions.base import CorrelatedNormal
from mutinfo.estimators.asmi import aSMI, AlignedRandomSlicing
from mutinfo.estimators.smi import SMI
from mutinfo.estimators.knn import KSG

from . import estimator_tester


def _backbone():
    return KSG(k_neighbors=5)


def test_asmi_recovers_mi_1d():
    """
    On 1D-1D data with projection_dim=1, the random "projection" is a 1x1
    orthogonal matrix (i.e. a sign flip) followed by mean-subtraction.
    Neither operation changes mutual information, so aSMI should recover
    the true MI within the same tolerance as the underlying KSG estimator.
    """
    estimator_tester.run_tests(
        lambda: aSMI(_backbone(), projection_dim=1, n_projection_samples=8),
        CorrelatedNormal,
        numpy.linspace(0.0, 1.0, 5),
        "Bad aSMI estimates on 1D-1D correlated Normal",
        [1], [1],
        n_samples=10000, atol=0.1, rtol=0.1,
    )


def test_asmi_vanishes_under_independence():
    """When X and Y are independent, aSMI must be approximately zero."""
    rng = numpy.random.RandomState(0)
    x = rng.randn(5000, 3)
    y = rng.randn(5000, 3)

    value = aSMI(_backbone(), projection_dim=1, n_projection_samples=64)(x, y)
    assert value < 0.05, f"aSMI on independent (X, Y) should be ~0; got {value}"


def test_aligned_random_slicing_shares_projector_when_same_dim():
    """
    When X and Y share their flattened ambient dimensionality,
    `AlignedRandomSlicing.fit` must reuse one projection matrix.
    """
    rng = numpy.random.RandomState(1)
    x = rng.randn(100, 4)
    y = rng.randn(100, 4)

    transform = AlignedRandomSlicing(projection_dim=2)
    transform.fit((x, y))

    assert transform.projector_x_ is transform.projector_y_, \
        "Aligned slicing should share the projector when ambient dims match"
    assert transform.projector_x_.shape == (4, 2)


def test_aligned_random_slicing_falls_back_when_dims_differ():
    """
    When X and Y have different flattened ambient dimensionalities,
    `AlignedRandomSlicing.fit` must fall back to independent matrices
    (otherwise the matmul on the smaller side would be ill-defined).
    """
    rng = numpy.random.RandomState(2)
    x = rng.randn(100, 3)
    y = rng.randn(100, 5)

    transform = AlignedRandomSlicing(projection_dim=1)
    transform.fit((x, y))

    assert transform.projector_x_ is not transform.projector_y_, \
        "Aligned slicing must use independent projectors when ambient dims differ"
    assert transform.projector_x_.shape == (3, 1)
    assert transform.projector_y_.shape == (5, 1)


def test_asmi_runs_on_different_ambient_dims():
    """aSMI must remain well-defined when X and Y have different ambient dims."""
    rng = numpy.random.RandomState(3)
    x = rng.randn(2000, 3)
    y = rng.randn(2000, 5)

    value = aSMI(_backbone(), projection_dim=1, n_projection_samples=8)(x, y)
    assert numpy.isfinite(value), f"aSMI must return a finite value; got {value}"


def test_asmi_beats_smi_on_diagonal_coupling():
    """
    Geometric advantage of aligned slicing.

    On a diagonal-coupled correlated Gaussian (Y_i correlated with X_i for
    each i), the dependence structure is aligned with the canonical basis.
    A shared random orthogonal projection rotates X and Y the same way, so
    the post-projection components remain pairwise correlated; under
    independent projections the matching is lost almost surely. Therefore
    aSMI should systematically exceed SMI in this regime.
    """
    rng = numpy.random.RandomState(4)
    d, n = 5, 4000
    rho = numpy.linspace(0.3, 0.95, d)
    z_x = rng.randn(n, d)
    z_y = rng.randn(n, d)
    x = z_x
    y = rho * z_x + numpy.sqrt(1.0 - rho ** 2) * z_y

    smi_value  = SMI(_backbone(),  projection_dim=1, n_projection_samples=128)(x, y)
    asmi_value = aSMI(_backbone(), projection_dim=1, n_projection_samples=128)(x, y)

    assert asmi_value > smi_value, \
        f"Expected aSMI > SMI on diagonal coupling; got aSMI={asmi_value}, SMI={smi_value}"