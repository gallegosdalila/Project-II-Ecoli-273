# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# tests/test_fields.py -- checks the three concentration profiles in fields.py.

# THE ONE THAT MATTERS MOST is test_analytical_gradient_matches_finite_difference.
    # gradient() returns a formula we worked out on paper. numerical_gradient() estimates the same thing by poking C either side and subtracting.
    # They are two COMPLETELY INDEPENDENT routes to the same number, so if they agree to six decimals the hand calculus is right.
    # This matters beyond fields.py: simulate.py's estimate_gradient is validated against gradient(), so if gradient() were wrong, that test would pass while the physics was silently broken.

import numpy as np
import pytest

from ecoli.fields import (CompetingSources, GaussianSource, LinearGradient,
                          make_field, FIELDS)


# The same three fields reused by the parametrized tests below, so each one gets checked against all three profiles.
ALL_FIELDS = [LinearGradient(slope=(0.02, -0.01)),
              GaussianSource(center=(10.0, -5.0), amplitude=80.0, sigma=40.0),
              CompetingSources()]


# ---------------------------------------------------------------------------
# The shape contract: (..., 2) in, (...) or (..., 2) out
# ---------------------------------------------------------------------------

# Every field must handle ONE point, an (N, 2) array of bacteria, and a (rows, cols, 2) plotting grid, all through the same code.
# parametrize runs this once per field in ALL_FIELDS, so one function covers all three.
def test_accepts_single_point_and_array(field):
    one = field.concentration(np.array([1.0, 2.0]))
    assert np.ndim(one) == 0                              # a single point gives back a single number, not an array of length 1
    many = field.concentration(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert many.shape == (2,)                             # (2, 2) in --> (2,) out, one concentration per bacterium
    grid = field.concentration(np.zeros((7, 5, 2)))
    assert grid.shape == (7, 5)                           # a plotting grid keeps its leading shape and loses only the trailing 2
test_accepts_single_point_and_array = pytest.mark.parametrize("field", ALL_FIELDS, ids=lambda f: f.name)(test_accepts_single_point_and_array)


# gradient() must return a VECTOR per point, so it keeps the trailing 2 that concentration() drops.
def test_gradient_shape_matches_input(field):
    p = np.zeros((6, 3, 2))
    assert field.gradient(p).shape == (6, 3, 2)
test_gradient_shape_matches_input = pytest.mark.parametrize("field", ALL_FIELDS, ids=lambda f: f.name)(test_gradient_shape_matches_input)


# ---------------------------------------------------------------------------
# THE CORE TEST: hand-derived calculus vs numerical estimate
# ---------------------------------------------------------------------------

# Checked at 200 random points rather than one, so a formula that happens to be right at the origin cannot sneak through.
# atol and rtol are loose-ish because numerical_gradient is itself an approximation --> agreement to ~1e-6 is as good as a finite difference gets.
def test_analytical_gradient_matches_finite_difference(field):
    rng = np.random.default_rng(0)
    p = rng.uniform(-80, 80, size=(200, 2))
    np.testing.assert_allclose(field.gradient(p), field.numerical_gradient(p, h=1e-4), atol=1e-6, rtol=1e-4)
test_analytical_gradient_matches_finite_difference = pytest.mark.parametrize("field", ALL_FIELDS, ids=lambda f: f.name)(test_analytical_gradient_matches_finite_difference)


# ---------------------------------------------------------------------------
# LinearGradient: the validation field, checkable on paper
# ---------------------------------------------------------------------------

# The whole point of the linear field is that its gradient is the SAME VECTOR EVERYWHERE, which is what makes it the anchor for simulate.py's tests.
def test_linear_gradient_is_constant_everywhere():
    f = LinearGradient(slope=(0.3, -0.7))
    rng = np.random.default_rng(1)
    g = f.gradient(rng.uniform(-1e3, 1e3, size=(500, 2)))   # deliberately huge coordinates: if the gradient depended on position at all, it would show up here
    assert g.shape == (500, 2)
    np.testing.assert_allclose(g, np.tile([0.3, -0.7], (500, 1)), atol=1e-12)


# THE HAND CHECK. With slope=(2,3), c0=5, reference=(1,1):
    # C(1,1) = 5              --> at the reference point, C is exactly c0
    # C(2,1) = 5 + 1*2 + 0*3  --> 7.0
    # C(1,2) = 5 + 0*2 + 1*3  --> 8.0
# These are done in your head, which is why this field is written before any curved one.
def test_linear_concentration_formula():
    f = LinearGradient(slope=(2.0, 3.0), c0=5.0, reference=(1.0, 1.0))
    assert f.concentration(np.array([1.0, 1.0])) == pytest.approx(5.0)
    assert f.concentration(np.array([2.0, 1.0])) == pytest.approx(7.0)
    assert f.concentration(np.array([1.0, 2.0])) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# GaussianSource: the main field for the headline result
# ---------------------------------------------------------------------------

# The peak is at the center and nowhere else is higher --> if this failed, "bacteria climb toward the source" would not even be well defined.
def test_gaussian_peaks_at_its_center():
    f = GaussianSource(center=(3.0, -4.0), amplitude=50.0, sigma=20.0)
    rng = np.random.default_rng(2)
    p = rng.uniform(-100, 100, size=(1000, 2))
    assert f.concentration(np.array([3.0, -4.0])) == pytest.approx(50.0)   # exactly the amplitude, at the center
    assert np.all(f.concentration(p) <= 50.0 + 1e-9)                        # nowhere higher, at 1000 random points


# THE TEST THAT JUSTIFIES ASCENT. grad C must point INWARD, back toward the source, at every point.
# We check it by taking the dot product of the normalized gradient with the unit vector pointing from the bacterium to the source: if they are the same direction, the dot product is exactly 1.
def test_gaussian_gradient_points_toward_the_source():
    f = GaussianSource(center=(0.0, 0.0), sigma=40.0)
    rng = np.random.default_rng(3)
    p = rng.uniform(-60, 60, size=(500, 2))
    p = p[np.linalg.norm(p, axis=1) > 1.0]                          # drop points sitting essentially ON the source, where the gradient is ~0 and has no meaningful direction
    inward = -p / np.linalg.norm(p, axis=1, keepdims=True)          # unit vector from each point back toward the origin
    g = f.gradient(p)
    g_hat = g / np.linalg.norm(g, axis=1, keepdims=True)
    np.testing.assert_allclose(np.einsum("ij,ij->i", g_hat, inward), 1.0, atol=1e-9)   # dot product of two unit vectors is 1 only when they point the same way


# ---------------------------------------------------------------------------
# CompetingSources: validated against the parts it is made of
# ---------------------------------------------------------------------------

# This field has no independent right answer to check against, so instead we confirm it equals the SUM of GaussianSources we already trust.
# Concentrations add because the sources are independent; gradients add because differentiation is linear.
def test_competing_sources_is_the_sum_of_its_parts():
    centers, amps, sigmas = ((-50.0, 0.0), (50.0, 0.0)), (100.0, 60.0), (30.0, 30.0)
    combo = CompetingSources(centers, amps, sigmas)
    parts = [GaussianSource(c, a, s) for c, a, s in zip(centers, amps, sigmas)]
    rng = np.random.default_rng(4)
    p = rng.uniform(-100, 100, size=(300, 2))
    np.testing.assert_allclose(combo.concentration(p), sum(f.concentration(p) for f in parts))
    np.testing.assert_allclose(combo.gradient(p), sum(f.gradient(p) for f in parts))


# There must be a SADDLE somewhere on the axis between the two peaks: a point where the x-gradient changes sign.
# That saddle is the watershed that splits the population, which is the whole reason this field is interesting.
def test_competing_sources_saddle_between_peaks():
    f = CompetingSources()
    x = np.linspace(-50, 50, 401)
    gx = f.gradient(np.stack((x, np.zeros_like(x)), axis=-1))[:, 0]   # the x component of the gradient, walking along the line y = 0
    assert np.any(gx > 0) and np.any(gx < 0)                          # pulled right somewhere, pulled left somewhere else --> a sign change in between


# ---------------------------------------------------------------------------
# Helpers and error handling
# ---------------------------------------------------------------------------

# meshgrid feeds plotting.py, so its three outputs must all be the same (n, n) shape that contourf expects.
def test_meshgrid_helper_shapes():
    X, Y, C = GaussianSource().meshgrid(half_width=50.0, n=64)
    assert X.shape == Y.shape == C.shape == (64, 64)


# Every name in the registry must build a working field, and a typo must fail loudly rather than silently.
def test_registry_builds_every_named_field():
    for name in FIELDS:
        f = make_field(name)
        assert np.atleast_2d(f.sources).shape[1] == 2   # every field reports its sources as (S, 2)
    with pytest.raises(KeyError):
        make_field("not_a_field")


# Bad input must be caught by _as_points at the door, not become a confusing broadcasting error deep in the math.
def test_bad_input_shape_raises():
    with pytest.raises(ValueError):
        GaussianSource().concentration(np.zeros((4, 3)))   # last axis is 3, not 2
