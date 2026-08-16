# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy
#
# tests/test_plotting.py -- Section 5. Owner: Nisa.
# Plotting tests check that figures BUILD and that the plotting layer never
# secretly runs a simulation. They do not check that a figure looks nice.

from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ecoli import plotting as P
from ecoli.config import Config
from ecoli.experiments import ExperimentRunner
from ecoli.fields import CompetingSources, GaussianSource, make_field
from ecoli.simulate import run_simulation, run_unbiased_control


CFG = Config(n_cells=60, n_iterations=40, start_mode="uniform",
             store_substeps=True)


# fixture: pytest builds this once and passes it into any test that names `result` as an argument.
def result():
    return run_simulation(GaussianSource(sigma=60.0), CFG,
                          np.random.default_rng(0))
result = pytest.fixture(scope="module")(result)


# fixture: pytest builds this once and passes it into any test that names `close_figures` as an argument.
def close_figures():
    yield
    plt.close("all")
close_figures = pytest.fixture(autouse=True)(close_figures)


def test_plot_concentration_builds_for_every_field():
    for name in ("shallow", "single_source", "competing_sources"):
        ax = P.plot_concentration(make_field(name), n=40)
        assert ax.collections, f"{name} drew nothing"


def test_plot_trajectories_marks_start_and_end(result):
    ax = P.plot_trajectories(result, n_show=10,
                             field=GaussianSource(sigma=60.0))
    labels = [c.get_label() for c in ax.collections if c.get_label()]
    assert "start" in labels and "end" in labels


def test_plot_trajectories_works_without_substeps():
    cfg = replace(CFG, store_substeps=False)
    r = run_simulation(GaussianSource(), cfg, np.random.default_rng(0))
    assert r.substeps is None
    assert P.plot_trajectories(r, n_show=5) is not None


def test_plot_trajectories_handles_n_show_greater_than_N(result):
    assert P.plot_trajectories(result, n_show=10_000) is not None


def test_plot_histograms_draws_one_curve_per_iteration(result):
    ax = P.plot_histograms(result, iterations=(1, 10, 40))
    assert len(ax.get_legend().get_texts()) == 3


def test_plot_histograms_ignores_out_of_range_iterations(result):
    ax = P.plot_histograms(result, iterations=(1, 10, 99999))
    assert len(ax.get_legend().get_texts()) == 2


def test_compare_N_makes_one_panel_per_population():
    field = GaussianSource(sigma=60.0)
    results = {n: run_simulation(field, replace(CFG, n_cells=n),
                                 np.random.default_rng(0))
               for n in (10, 50)}
    fig, axes = P.compare_N(results)
    assert len(axes) == 2
    assert axes[0].get_title() == "N = 10"


def test_plot_convergence_labels_each_N():
    field = GaussianSource(sigma=60.0)
    results = {n: run_simulation(field, replace(CFG, n_cells=n),
                                 np.random.default_rng(0))
               for n in (10, 50)}
    ax = P.plot_convergence(results)
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {
        "N = 10", "N = 50"}


def test_plot_biased_vs_control_builds(result):
    control = run_unbiased_control(GaussianSource(sigma=60.0), CFG,
                                   np.random.default_rng(0))
    ax = P.plot_biased_vs_control(result, control)
    assert len(ax.get_lines()) == 2


def test_plotting_never_runs_a_simulation(monkeypatch, result):
    """Guard rail for the Section 5 contract."""
    import ecoli.simulate as sim
    monkeypatch.setattr(sim, "run_simulation", lambda *a, **k:
                        pytest.fail("plotting called run_simulation"))
    P.plot_histograms(result, iterations=(1, 10))
    P.plot_trajectories(result, n_show=5)


# ---------------------------------------------------------------------------
# Full integration: simulate -> save -> load -> every figure on disk
# ---------------------------------------------------------------------------

def test_end_to_end_figure_regeneration(tmp_path):
    from ecoli import figures

    cfg = replace(CFG, n_iterations=30)
    runner = ExperimentRunner("single_source", cfg, tmp_path / "results")
    for n in (10, 100, 1000):
        runner.run_one(n, seed=0, biased=True)
    runner.run_one(1000, seed=0, biased=False)

    made = figures.build_all(results_dir=tmp_path / "results",
                             fig_dir=tmp_path / "figures",
                             field_name="single_source", seed=0)
    assert len(made) >= 6
    for path in made:
        assert path.stat().st_size > 5_000, f"{path.name} looks empty"


def test_figures_raise_when_results_are_missing(tmp_path):
    from ecoli import figures
    with pytest.raises(FileNotFoundError):
        figures.figure_trajectories(tmp_path, tmp_path, "single_source")
