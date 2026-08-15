# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy
#
# tests/test_experiments.py -- Section 4. Owner: Aleyna.

from dataclasses import replace

import numpy as np
import pytest

from ecoli.config import Config
from ecoli.contracts import SimulationResult
from ecoli.experiments import ExperimentRunner, result_path
from ecoli.fields import GaussianSource
from ecoli.simulate import run_simulation
from ecoli.stats import (chemotactic_drift, distance_stats, distance_trace,
                         fraction_within, mean_squared_displacement,
                         snapshot_stats)


BASE = Config(n_iterations=80, start_mode="uniform")


# fixture: pytest builds this once and passes it into any test that names `result` as an argument.
def result():
    cfg = replace(BASE, n_cells=300)
    return run_simulation(GaussianSource(sigma=60.0), cfg,
                          np.random.default_rng(0))
result = pytest.fixture(scope="module")(result)


# ---------------------------------------------------------------------------
# stats.py
# ---------------------------------------------------------------------------

def test_distance_stats_on_a_known_array():
    s = distance_stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(2.5)
    assert s["median"] == pytest.approx(2.5)
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert s["variance"] == pytest.approx(np.var([1, 2, 3, 4], ddof=1))
    assert s["std"] == pytest.approx(np.sqrt(s["variance"]))


def test_distance_stats_handles_single_cell():
    s = distance_stats(np.array([7.0]))
    assert s["std"] == 0.0 and s["variance"] == 0.0


def test_distance_trace_lengths(result):
    trace = distance_trace(result)
    for key in ("mean", "median", "std", "sem"):
        assert trace[key].shape == (result.n_iterations + 1,)
    np.testing.assert_allclose(
        trace["sem"], trace["std"] / np.sqrt(result.n_cells))


def test_snapshot_stats_skips_out_of_range_iterations(result):
    out = snapshot_stats(result, (1, 10, 50, 100, 1000))
    assert set(out) == {1, 10, 50}          # run is only 80 cycles long
    assert out[50]["mean"] < out[1]["mean"]


def test_msd_is_monotonic_and_right_length(result):
    msd = mean_squared_displacement(result)
    assert msd.shape == (result.n_iterations,)
    assert msd[0] < msd[-1]


def test_chemotactic_drift_positive_when_converging(result):
    assert chemotactic_drift(result) > 0


def test_fraction_within_is_a_probability(result):
    assert fraction_within(result, radius=1e9) == 1.0
    assert fraction_within(result, radius=0.0) < 1.0


# ---------------------------------------------------------------------------
# experiments.py
# ---------------------------------------------------------------------------

def test_runner_sweeps_population_sizes(tmp_path):
    runner = ExperimentRunner("single_source", replace(BASE, n_iterations=20),
                              tmp_path)
    out = runner.sweep_population(sizes=(10, 50), seed=0)
    assert sorted(out) == [10, 50]
    assert out[10].n_cells == 10 and out[50].n_cells == 50
    for n in (10, 50):
        assert result_path(tmp_path, "single_source", n, 0).exists()


def test_saved_results_reload_identically(tmp_path):
    runner = ExperimentRunner("single_source", replace(BASE, n_iterations=15),
                              tmp_path)
    original = runner.run_one(20, seed=3)
    back = SimulationResult.load(
        result_path(tmp_path, "single_source", 20, 3))
    np.testing.assert_array_equal(original.positions, back.positions)


def test_seed_sweep_gives_different_but_valid_runs(tmp_path):
    runner = ExperimentRunner("single_source", replace(BASE, n_iterations=15),
                              tmp_path)
    runs = runner.sweep_seeds(30, seeds=(0, 1, 2))
    assert len(runs) == 3
    assert not np.allclose(runs[0].positions, runs[1].positions)
    for r in runs:
        r.validate()


def test_larger_N_tightens_the_estimate_of_the_mean():
    """Section 4 half of the 'narrowing' claim.

    The population mean distance is a sample mean, so its standard error
    falls like 1/sqrt(N). Across N = 10, 100, 1000 the SEM must drop
    monotonically even though the underlying distribution is unchanged.
    """
    field = GaussianSource(sigma=60.0)
    sems = [
        distance_trace(run_simulation(field, replace(BASE, n_cells=n),
                                      np.random.default_rng(0)))["sem"][-1]
        for n in (10, 100, 1000)
    ]
    assert sems[0] > sems[1] > sems[2]


def test_load_all_recovers_the_sweep(tmp_path):
    runner = ExperimentRunner("single_source", replace(BASE, n_iterations=10),
                              tmp_path)
    runner.sweep_population(sizes=(10, 40), seed=0)
    loaded = runner.load_all()
    assert sorted(loaded) == [10, 40]
