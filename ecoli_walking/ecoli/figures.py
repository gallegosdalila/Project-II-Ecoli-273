# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# figures.py -- regenerates EVERY figure in the report, from SAVED results, with one command:
#     python -m ecoli.figures --results results --figures figures

# WHY THIS FILE EXISTS SEPARATELY FROM plotting.py:
    # plotting.py knows HOW to draw one figure onto one axes. It has no idea which figures the report needs or where the data lives.
    # figures.py knows WHICH figures the report needs, which saved file each one comes from, and what to name the output. It draws nothing itself.
    # Splitting them means a figure can be restyled without touching the report layout, and the report can gain a figure without touching any drawing code.

# If a needed .npz is MISSING this raises instead of quietly running a simulation, so a figure can never silently disagree with the numbers in the results table.

# THIS FILE IS PHASE 7 -- step 36 of the build order.

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # the non-interactive backend. MUST be set BEFORE pyplot is imported, or matplotlib picks a backend that needs a screen and crashes on a machine without one.
import matplotlib.pyplot as plt
import numpy as np

from .config import POPULATION_SIZES, SNAPSHOT_ITERATIONS
from .contracts import SimulationResult
from .experiments import result_path
from .fields import make_field
from . import plotting as P


# Load ONE saved run, or fail with the exact command needed to produce it. Never simulates as a fallback.
def _load(results_dir, field_name, n_cells, seed, biased=True):
    path = result_path(results_dir, field_name, n_cells, seed, biased)   # the same naming rule experiments.py used to write the file, so the two can never disagree
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the experiment first:\n    python -m ecoli.main --field {field_name} --n {n_cells} --seed {seed}")
    return SimulationResult.load(path)


# FIGURE 1: the three required concentration profiles side by side. The only figure that needs no saved results, since it draws the fields themselves.
def figure_fields(fig_dir: Path, half_width: float = 100.0):
    names = ["shallow", "single_source", "competing_sources"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, name in zip(axes, names):
        P.plot_concentration(make_field(name), half_width=half_width, ax=ax, colorbar=False)
    fig.suptitle("Concentration profiles")
    fig.savefig(fig_dir / "fig1_fields.png")
    plt.close(fig)   # close EXPLICITLY, or matplotlib keeps every figure in memory and warns once past 20 open at once


# FIGURE 2: sample paths drawn over the field, start and end marked.
# Needs a run saved with store_substeps=True, or the paths come out as plain zigzags with no visible tumbling.
def figure_trajectories(results_dir, fig_dir: Path, field_name: str, n_cells: int = 100, seed: int = 0):
    r = _load(results_dir, field_name, n_cells, seed)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    P.plot_trajectories(r, n_show=20, field=make_field(field_name), ax=ax)
    fig.savefig(fig_dir / f"fig2_trajectories_{field_name}.png")
    plt.close(fig)


# FIGURE 3: distance histograms tightening as the iteration count grows. THIS IS THE HEADLINE RESULT.
def figure_narrowing(results_dir, fig_dir: Path, field_name: str, n_cells: int = 1000, seed: int = 0):
    r = _load(results_dir, field_name, n_cells, seed)
    iters = [i for i in SNAPSHOT_ITERATIONS if i <= r.n_iterations]   # SNAPSHOT_ITERATIONS lives in config.py, so the figure and the results table show the SAME iteration counts
    fig, ax = plt.subplots()
    P.plot_histograms(r, iterations=iters, ax=ax)
    fig.savefig(fig_dir / f"fig3_narrowing_{field_name}.png")
    plt.close(fig)


# FIGURES 4 and 5: N = 10 / 100 / 1000 histograms at the final iteration, then the convergence curves for those same three runs.
# Two figures from one set of loaded results, because loading the N=1000 file twice would be wasteful.
def figure_population_sweep(results_dir, fig_dir: Path, field_name: str, seed: int = 0):
    results = {n: _load(results_dir, field_name, n, seed) for n in POPULATION_SIZES}

    fig, _ = P.compare_N(results)
    fig.savefig(fig_dir / f"fig4_compare_N_{field_name}.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    P.plot_convergence(results, ax=ax)
    fig.savefig(fig_dir / f"fig5_convergence_{field_name}.png")
    plt.close(fig)


# FIGURE 6: biased against the unbiased control. This is the panel that makes the whole result falsifiable.
def figure_control(results_dir, fig_dir: Path, field_name: str, n_cells: int = 1000, seed: int = 0):
    biased = _load(results_dir, field_name, n_cells, seed, biased=True)
    control = _load(results_dir, field_name, n_cells, seed, biased=False)
    fig, ax = plt.subplots()
    P.plot_biased_vs_control(biased, control, ax=ax)
    fig.savefig(fig_dir / f"fig6_control_{field_name}.png")
    plt.close(fig)


# Build every figure in order and return the list of files written, which is what the integration test checks.
def build_all(results_dir="results", fig_dir="figures", field_name="single_source", seed: int = 0):
    P.use_style()   # apply the shared house style ONCE, before anything is drawn --> every figure below inherits it
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_fields(fig_dir)
    figure_trajectories(results_dir, fig_dir, field_name, seed=seed)
    figure_narrowing(results_dir, fig_dir, field_name, seed=seed)
    figure_population_sweep(results_dir, fig_dir, field_name, seed=seed)
    figure_control(results_dir, fig_dir, field_name, seed=seed)
    return sorted(fig_dir.glob("*.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate all report figures from saved results.")
    ap.add_argument("--results", default="results", help="folder holding the .npz files written by main.py")
    ap.add_argument("--figures", default="figures", help="folder to write the .png files into")
    ap.add_argument("--field", default="single_source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    made = build_all(args.results, args.figures, args.field, args.seed)
    print("\n".join(str(p) for p in made))


if __name__ == "__main__":   # only runs when this file is executed directly, not when another file imports it
    main()
