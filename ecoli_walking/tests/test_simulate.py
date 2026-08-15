# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy
#
# tests/test_simulate.py -- Section 3. Owner: Tracy.
# The two tests that matter most for the whole project are
# test_estimate_matches_analytical_gradient_on_linear_field and
# test_biased_beats_unbiased_control. If those pass, the physics is right.

from dataclasses import replace

import numpy as np
import pytest

from ecoli.config import Config
from ecoli.fields import GaussianSource, LinearGradient, CompetingSources
from ecoli.simulate import (choose_run_direction, directional_derivative,
                            estimate_gradient, run_simulation,
                            run_unbiased_control)


CFG = Config(n_cells=200, n_iterations=60, start_mode="uniform", seed=0)


# ---------------------------------------------------------------------------
# estimate_gradient, the finite-difference core
# ---------------------------------------------------------------------------

def test_estimate_recovers_the_directional_derivative():
    """g_est . d must equal delta_c exactly, by construction."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 2))
    dc = rng.normal(size=500)
    g, valid = estimate_gradient(dc, d)
    assert valid.all()
    np.testing.assert_allclose(np.einsum("ij,ij->i", g, d), dc, atol=1e-9)


def test_estimate_matches_analytical_gradient_on_linear_field():
    """On a LINEAR field the estimate is the exact projection of grad C.

    g_est is the component of the true gradient along the sampled
    direction, so g_est should equal (grad C . d_hat) d_hat.
    """
    field = LinearGradient(slope=(0.03, -0.02))
    rng = np.random.default_rng(1)
    x0 = rng.uniform(-80, 80, size=(1000, 2))
    d = rng.normal(scale=2.0, size=(1000, 2))
    dc = field.concentration(x0 + d) - field.concentration(x0)

    g_est, _ = estimate_gradient(dc, d)
    d_hat = d / np.linalg.norm(d, axis=1, keepdims=True)
    true_g = field.gradient(x0)
    projection = np.einsum("ij,ij->i", true_g, d_hat)[:, None] * d_hat
    np.testing.assert_allclose(g_est, projection, atol=1e-10)


def test_estimate_sign_is_correct():
    """Uphill sampling gives a positive dC and an outward-pointing estimate."""
    field = LinearGradient(slope=(1.0, 0.0))
    x0 = np.zeros((3, 2))
    d = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    dc = field.concentration(x0 + d) - field.concentration(x0)
    assert dc[0] > 0 and dc[1] < 0 and dc[2] == pytest.approx(0.0)
    g, _ = estimate_gradient(dc, d)
    assert g[0, 0] > 0            # moved uphill -> keep going +x
    assert g[1, 0] > 0            # moved downhill -> estimate still points +x


def test_near_zero_displacement_is_flagged_not_nan():
    dc = np.array([1.0, 1.0])
    d = np.array([[1.0, 0.0], [1e-15, 0.0]])
    g, valid = estimate_gradient(dc, d, min_displacement=1e-9)
    assert valid.tolist() == [True, False]
    assert np.isfinite(g).all()
    np.testing.assert_allclose(g[1], [0.0, 0.0])


def test_directional_derivative_is_dc_over_distance():
    dc = np.array([2.0, 0.0])
    d = np.array([[4.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(directional_derivative(dc, d), [0.5, 0.0])


# ---------------------------------------------------------------------------
# choose_run_direction, including the fallback branches
# ---------------------------------------------------------------------------

def test_ascent_and_descent_are_opposite():
    rng = np.random.default_rng(2)
    g = rng.normal(size=(50, 2))
    valid = np.ones(50, dtype=bool)
    prev = np.zeros((50, 2))
    up = choose_run_direction(g, valid, prev, rng, replace(CFG, ascend=True))
    down = choose_run_direction(g, valid, prev, rng, replace(CFG, ascend=False))
    np.testing.assert_allclose(up, -down, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(up, axis=1), 1.0, atol=1e-12)


# parametrize runs this once per fallback mode, so all three branches are covered by one function.
def test_fallback_modes(mode, expect_move):
    rng = np.random.default_rng(3)
    g = np.zeros((10, 2))
    valid = np.zeros(10, dtype=bool)
    prev = np.tile([1.0, 0.0], (10, 1))
    out = choose_run_direction(g, valid, prev, rng, replace(CFG, fallback=mode))
    moved = np.linalg.norm(out, axis=1) > 0
    assert moved.all() == expect_move
test_fallback_modes = pytest.mark.parametrize("mode,expect_move", [("previous", True), ("random", True), ("skip", False)])(test_fallback_modes)


# ---------------------------------------------------------------------------
# Whole-simulation behaviour
# ---------------------------------------------------------------------------

def test_result_shapes_and_validation():
    cfg = replace(CFG, n_cells=25, n_iterations=17, store_substeps=True)
    r = run_simulation(GaussianSource(), cfg, np.random.default_rng(0))
    r.validate()
    assert r.positions.shape == (25, 18, 2)
    assert r.concentrations.shape == (25, 18)
    assert r.run_directions.shape == (25, 17, 2)
    assert r.substeps.shape == (25, 17, cfg.n_tumbles + 2, 2)
    assert r.n_cells == 25 and r.n_iterations == 17


def test_simulation_is_reproducible_with_the_same_seed():
    a = run_simulation(GaussianSource(), CFG, np.random.default_rng(42))
    b = run_simulation(GaussianSource(), CFG, np.random.default_rng(42))
    np.testing.assert_array_equal(a.positions, b.positions)


def test_biased_population_drifts_up_a_linear_gradient():
    """Net displacement should align with the constant gradient direction."""
    slope = np.array([0.02, 0.01])
    field = LinearGradient(slope=tuple(slope))
    r = run_simulation(field, CFG, np.random.default_rng(0))
    mean_disp = r.net_displacement().mean(axis=0)
    cos = (mean_disp @ slope) / (np.linalg.norm(mean_disp) *
                                 np.linalg.norm(slope))
    assert cos > 0.9, f"population drifted off-gradient, cos = {cos:.3f}"


def test_biased_beats_unbiased_control():
    """The headline validation: information helps."""
    field = GaussianSource(sigma=60.0)
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    biased = run_simulation(field, CFG, rng_a)
    control = run_unbiased_control(field, CFG, rng_b)
    assert (biased.distances_to_source()[:, -1].mean()
            < control.distances_to_source()[:, -1].mean())
    assert (biased.concentrations[:, -1].mean()
            > control.concentrations[:, -1].mean())


def test_unbiased_control_has_no_net_drift():
    field = LinearGradient(slope=(0.02, 0.01))
    cfg = replace(CFG, n_cells=2000)
    control = run_unbiased_control(field, cfg, np.random.default_rng(1))
    drift = np.linalg.norm(control.net_displacement().mean(axis=0))
    expected_random = cfg.run_length * np.sqrt(cfg.n_iterations / cfg.n_cells)
    assert drift < 5 * expected_random


def test_descent_moves_cells_away_from_the_source():
    field = GaussianSource(sigma=60.0)
    cfg = replace(CFG, ascend=False)
    r = run_simulation(field, cfg, np.random.default_rng(0))
    d = r.distances_to_source()
    assert d[:, -1].mean() > d[:, 0].mean()


def test_more_iterations_tightens_the_distance_distribution():
    """Section 3 half of the 'narrowing' claim: larger I, smaller spread."""
    field = GaussianSource(sigma=60.0)
    cfg = replace(CFG, n_cells=500, n_iterations=300)
    r = run_simulation(field, cfg, np.random.default_rng(0))
    d = r.distances_to_source()
    early, late = d[:, 10], d[:, -1]
    assert late.mean() < early.mean()
    assert late.std() < early.std()


def test_competing_sources_splits_the_population():
    field = CompetingSources()
    cfg = replace(CFG, n_cells=400, n_iterations=200)
    r = run_simulation(field, cfg, np.random.default_rng(0))
    final = r.positions[:, -1, :]
    left = (final[:, 0] < 0).mean()
    assert 0.05 < left < 0.95, "everyone went to the same peak"


def test_gradient_valid_is_almost_always_true():
    """Four random unit steps almost never cancel exactly."""
    r = run_simulation(GaussianSource(), CFG, np.random.default_rng(0))
    assert r.gradient_valid.mean() > 0.99


def test_boundary_reflect_keeps_everyone_inside():
    cfg = replace(CFG, boundary="reflect", domain_half_width=50.0)
    r = run_simulation(GaussianSource(), cfg, np.random.default_rng(0))
    assert np.all(np.abs(r.positions) <= 50.0 + 1e-9)


def test_save_and_load_roundtrip(tmp_path):
    cfg = replace(CFG, n_cells=8, n_iterations=6, store_substeps=True)
    r = run_simulation(GaussianSource(), cfg, np.random.default_rng(0))
    path = tmp_path / "r.npz"
    r.save(path)
    from ecoli.contracts import SimulationResult
    back = SimulationResult.load(path)
    back.validate()
    np.testing.assert_array_equal(r.positions, back.positions)
    np.testing.assert_array_equal(r.substeps, back.substeps)
    assert back.field_name == r.field_name
    assert back.config["n_cells"] == 8
