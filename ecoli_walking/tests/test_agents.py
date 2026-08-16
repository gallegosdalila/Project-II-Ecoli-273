# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# tests/test_agents.py -- checks the movement layer in agents.py.

# These four things are what the project spec asks this file to confirm:
    # direction vectors have length 1     --> test_random_unit_vectors_are_unit_length
    # step lengths are correct            --> test_tumble_step_length_is_exact, test_run_step_length_is_exact
    # fixed random seeds are reproducible --> test_same_seed_reproduces_exactly
    # the domain rules behave             --> test_boundary_keeps_cells_in_domain

# Everything here is GEOMETRY, not physics. agents.py never sees a concentration, so nothing in this file mentions one.
# That is exactly why these tests can be so strict: a step length either IS 5.0 um or it is not, with no statistics involved.

from dataclasses import replace

import numpy as np
import pytest

from ecoli.agents import (Population, apply_boundary, initial_positions,
                          normalize, random_unit_vectors)
from ecoli.config import Config


CFG = Config(n_cells=50, tumble_step=1.0, run_length=5.0)


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

# A "unit vector" means length exactly 1, and atol=1e-12 is machine precision --> this is not a statistical claim, it either holds or the trig is wrong.
def test_random_unit_vectors_are_unit_length():
    rng = np.random.default_rng(1)
    v = random_unit_vectors(rng, 1000)
    assert v.shape == (1000, 2)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)


# ISOTROPIC means no preferred direction. If the headings were biased, the average of many of them would point somewhere instead of cancelling to ~zero.
# This matters because a tumble that secretly favoured one direction would create drift with no gradient at all, and the whole result would be an artifact.
def test_random_directions_are_isotropic():
    rng = np.random.default_rng(2)
    v = random_unit_vectors(rng, 200_000)
    assert np.linalg.norm(v.mean(axis=0)) < 0.01   # 200k vectors averaging to almost nothing


# A zero row means "this bacterium has no direction this cycle", which is a VALID state, so it must stay zero rather than becoming NaN.
# If it became NaN it would spread into every position downstream and quietly poison every figure.
def test_normalize_leaves_zero_rows_zero_not_nan():
    v = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = normalize(v)
    np.testing.assert_allclose(out[0], [0.6, 0.8])   # the 3-4-5 triangle, checkable by hand
    np.testing.assert_allclose(out[1], [0.0, 0.0])
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Step lengths
# ---------------------------------------------------------------------------

# Every one of the 4 tumble steps must be exactly tumble_step long, for every bacterium.
# We reconstruct the individual steps with np.diff on the full path, which also checks that cumsum inside tumble() assembled them correctly.
def test_tumble_step_length_is_exact():
    rng = np.random.default_rng(3)
    pop = Population.from_config(CFG, rng)
    before = pop.positions.copy()
    path = pop.tumble(rng)
    assert path.shape == (CFG.n_cells, CFG.n_tumbles, 2)
    full = np.concatenate((before[:, None, :], path), axis=1)   # glue the starting position on the front, so diff gives all 4 steps
    lengths = np.linalg.norm(np.diff(full, axis=1), axis=-1)
    np.testing.assert_allclose(lengths, CFG.tumble_step, atol=1e-12)


# The directed run must move exactly run_length, which is what makes it beat the tumbles on average.
def test_run_step_length_is_exact():
    rng = np.random.default_rng(4)
    pop = Population.from_config(CFG, rng)
    before = pop.positions.copy()
    u = random_unit_vectors(rng, pop.n)
    pop.run(u)
    moved = np.linalg.norm(pop.positions - before, axis=-1)
    np.testing.assert_allclose(moved, CFG.run_length, atol=1e-12)


# A ZERO direction vector is how choose_run_direction says "skip this cycle", so run() must leave those bacteria exactly where they were.
def test_zero_direction_means_no_run():
    rng = np.random.default_rng(5)
    pop = Population.from_config(CFG, rng)
    before = pop.positions.copy()
    pop.run(np.zeros((pop.n, 2)))
    np.testing.assert_allclose(pop.positions, before)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# THE WHOLE POINT OF PASSING ONE GENERATOR DOWN: the same seed must give byte-identical results, or no figure in the report can be regenerated.
def test_same_seed_reproduces_exactly():
    a = Population.from_config(CFG, np.random.default_rng(7))
    b = Population.from_config(CFG, np.random.default_rng(7))
    a.tumble(np.random.default_rng(8))
    b.tumble(np.random.default_rng(8))
    np.testing.assert_array_equal(a.positions, b.positions)   # assert_array_equal, not allclose --> we want EXACTLY the same bits


# The other half of the claim: different seeds must actually give different runs, or "reproducible" would just mean the randomness is broken.
def test_different_seeds_differ():
    a = Population.from_config(CFG, np.random.default_rng(7))
    b = Population.from_config(CFG, np.random.default_rng(9))
    assert not np.allclose(a.positions, b.positions)


# ---------------------------------------------------------------------------
# Domain boundaries
# ---------------------------------------------------------------------------

# Both bounded rules must contain every bacterium, no matter how far outside it started.
# We throw positions from -500 to +500 at a domain of half-width 10, so the fold has to work many times over, not just once.
# parametrize runs this same test once per rule, so one function covers both.
def test_boundary_keeps_cells_in_domain(rule):
    cfg = replace(CFG, boundary=rule, domain_half_width=10.0)
    rng = np.random.default_rng(11)
    xy = rng.uniform(-500, 500, size=(5000, 2))
    out = apply_boundary(xy, cfg)
    assert np.all(np.abs(out) <= cfg.domain_half_width + 1e-9)
test_boundary_keeps_cells_in_domain = pytest.mark.parametrize("rule", ["wrap", "reflect"])(test_boundary_keeps_cells_in_domain)


# The default rule is "none", meaning unbounded, so it must leave positions completely untouched even a million um out.
def test_boundary_none_is_identity():
    xy = np.array([[1e6, -1e6]])
    np.testing.assert_array_equal(apply_boundary(xy, CFG), xy)


# ---------------------------------------------------------------------------
# Starting positions
# ---------------------------------------------------------------------------

# All three start modes must produce (N, 2), and "ring" must put every bacterium at exactly the same distance out.
def test_initial_positions_shapes_and_modes():
    rng = np.random.default_rng(12)
    for mode in ("uniform", "point", "ring"):
        cfg = replace(CFG, start_mode=mode)
        xy = initial_positions(rng, cfg)
        assert xy.shape == (cfg.n_cells, 2)
    ring = initial_positions(rng, replace(CFG, start_mode="ring", start_radius=25.0))
    np.testing.assert_allclose(np.linalg.norm(ring, axis=1), 25.0, atol=1e-12)


# "point" must stack every bacterium on the SAME spot --> this is the mode to avoid for the headline figure, since distance would start at 0 and could only grow.
def test_point_start_stacks_every_cell_together():
    rng = np.random.default_rng(13)
    xy = initial_positions(rng, replace(CFG, start_mode="point", start_point=(-70.0, -70.0)))
    np.testing.assert_allclose(xy, np.tile([-70.0, -70.0], (CFG.n_cells, 1)))


# A typo in a config string must fail loudly at the door rather than silently doing something unexpected.
def test_unknown_modes_raise():
    with pytest.raises(ValueError):
        initial_positions(np.random.default_rng(0), replace(CFG, start_mode="spiral"))
    with pytest.raises(ValueError):
        apply_boundary(np.zeros((3, 2)), replace(CFG, boundary="bounce"))
