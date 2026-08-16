# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# plotting.py -- every figure in the report, drawn from a finished run.

# THE ONE RULE FOR THIS FILE: NOTHING HERE RUNS A SIMULATION.
    # Every function takes a SimulationResult (or a field) that somebody else already produced, draws on an axes, and returns that axes so the caller can keep customizing it.
    # test_plotting.py has a test that FAILS if this file ever calls run_simulation, because if plotting could rerun the simulation then a figure and the results table could quietly come from two DIFFERENT runs and nobody would notice.
    # If a plotting function needs data it does not have, the fix belongs in experiments.py, not here.

# THIS FILE IS PHASE 7 -- steps 31 to 35 of the build order.
# It comes last because a figure of a broken simulation is just a prettier way to be wrong. Everything it draws has already been checked numerically by stats.py and the test suite.

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .contracts import SimulationResult
from .fields import ConcentrationField
from .stats import distance_trace


# ---------------------------------------------------------------------------
# Step 31: House Style And Shared Helpers
# ---------------------------------------------------------------------------

# ONE shared style dict, so every figure in the report matches without a separate .mplstyle file to lose track of.
STYLE = {
    "figure.figsize": (6.0, 5.0),
    "figure.dpi": 130,          # screen resolution while working
    "savefig.dpi": 300,         # print resolution for the report --> 300 is the usual minimum for a figure that will be printed
    "savefig.bbox": "tight",    # trims the whitespace matplotlib leaves around the axes, so figures do not float in a sea of margin
    "axes.grid": True,
    "grid.alpha": 0.25,         # faint enough that the grid never competes with the data
    "axes.spines.top": False,   # dropping the top and right box lines is standard for scientific figures --> less ink, same information
    "axes.spines.right": False,
    "font.size": 11,
    "legend.frameon": False,    # no box around the legend, same reasoning
}

SOURCE_MARKER = dict(marker="*", s=320, c="crimson", edgecolors="black", linewidths=0.6, zorder=6)   # zorder=6 keeps the source star drawn ON TOP of everything else, since it is the reference point for every figure


# Call this ONCE at the top of figures.py, before anything is drawn. rcParams is matplotlib's global settings dict.
def use_style() -> None:
    plt.rcParams.update(STYLE)


# Use the axes the caller handed us, or make a fresh figure if they did not.
# This is what lets every function below work either standalone (for quick checks) or as ONE PANEL of a bigger multi-panel figure, without writing each function twice.
def _ax(ax=None):
    return ax if ax is not None else plt.subplots()[1]   # plt.subplots() returns (figure, axes), so [1] takes just the axes


# ---------------------------------------------------------------------------
# Step 32: The Concentration Field
# ---------------------------------------------------------------------------

# Filled contour map of C(x, y) with the source(s) marked. This is the background every trajectory plot is drawn on top of.
    # half_width: how far out to draw, um --> should match cfg.domain_half_width or the figure will not line up with where the bacteria actually are
    # n:          grid resolution. 300 x 300 is smooth; drop to 40 in tests to keep them fast.
def plot_concentration(field: ConcentrationField, half_width: float = 100.0, n: int = 300,
                       ax=None, levels: int = 25, colorbar: bool = True):
    ax = _ax(ax)
    X, Y, C = field.meshgrid(half_width, n)                                         # the grid comes from fields.py, so the figure and the simulation are guaranteed to read the SAME C(x, y)
    cf = ax.contourf(X, Y, C, levels=levels, cmap="viridis", alpha=0.9)             # viridis is perceptually uniform and still readable in greyscale, unlike the old jet colormap which invents bands that are not in the data
    ax.contour(X, Y, C, levels=levels, colors="white", linewidths=0.3, alpha=0.4)   # faint white contour LINES on top of the fill make the level spacing legible
    src = np.atleast_2d(field.sources)
    ax.scatter(src[:, 0], src[:, 1], **SOURCE_MARKER)
    ax.set(xlabel="x (um)", ylabel="y (um)", aspect="equal", title=f"Concentration field: {field.name}")
    # aspect="equal" MATTERS: without it a circular Gaussian renders as an ellipse and distances look wrong in one direction
    if colorbar:
        ax.figure.colorbar(cf, ax=ax, label="C (arb. units)", shrink=0.85)   # skipped when this is one panel of a multi-panel figure, where three colorbars would just be clutter
    return ax


# ---------------------------------------------------------------------------
# Step 33: Trajectories
# ---------------------------------------------------------------------------

# Paths of a SUBSAMPLE of bacteria, with start and end marked, drawn over the field.
# Uses the full within-cycle path if the run stored substeps (which shows the run-and-tumble texture), otherwise falls back to cycle endpoints (which looks like a plain zigzag).
    # n_show: how many bacteria to draw. Drawing all 1000 is an unreadable ball of yarn --> 20 is enough to see the behaviour.
def plot_trajectories(result: SimulationResult, n_show: int = 20, field: Optional[ConcentrationField] = None,
                      ax=None, rng: Optional[np.random.Generator] = None):
    ax = _ax(ax)
    if field is not None:
        plot_concentration(field, ax=ax, colorbar=False)   # draw the field UNDERNEATH first, so the paths sit on top of it

    rng = rng or np.random.default_rng(0)
    k = min(n_show, result.n_cells)                        # min() so asking for 20 trajectories from a 10-bacterium run does not crash
    idx = rng.choice(result.n_cells, size=k, replace=False)   # replace=False so we never draw the same bacterium twice

    if result.substeps is not None:
        paths = result.substeps[idx].reshape(k, -1, 2)     # flatten the cycle axis and the sub-step axis into ONE long path per bacterium --> this is what shows the tumbles
    else:
        paths = result.positions[idx]

    # ONE LineCollection beats k separate calls to ax.plot once k is in the hundreds, because matplotlib draws it as a single artist instead of k of them.
    from matplotlib.collections import LineCollection
    segments = np.stack((paths[:, :-1, :], paths[:, 1:, :]), axis=2)   # pair each point with the NEXT one to make line segments --> [:-1] is "all but the last", [1:] is "all but the first"
    ax.add_collection(LineCollection(segments.reshape(-1, 2, 2), colors="white", linewidths=0.7, alpha=0.75, zorder=3))
    ax.scatter(paths[:, 0, 0], paths[:, 0, 1], s=22, c="white", edgecolors="black", linewidths=0.5, zorder=5, label="start")    # index 0 along the path axis
    ax.scatter(paths[:, -1, 0], paths[:, -1, 1], s=34, c="orange", edgecolors="black", linewidths=0.5, zorder=5, label="end")   # index -1 is the last position
    ax.set(xlabel="x (um)", ylabel="y (um)", aspect="equal", title=f"{k} trajectories, N={result.n_cells}, I={result.n_iterations}")
    ax.legend(loc="upper right")
    return ax


# ---------------------------------------------------------------------------
# Step 34: The Headline Figure
# ---------------------------------------------------------------------------

# Distance-from-source histograms at several iteration counts.
# THIS IS THE FIGURE THE WHOLE REPORT IS BUILT AROUND: the assignment asks how the distance distribution narrows as the iteration count grows, and this is the picture of it.
def plot_histograms(result: SimulationResult, iterations: Sequence[int] = (1, 10, 50, 100),
                    bins: int = 30, ax=None, nearest: bool = False):
    ax = _ax(ax)
    d = result.distances_to_nearest_source() if nearest else result.distances_to_source()
    shown = [i for i in iterations if 0 <= i <= result.n_iterations]   # skip any requested iteration the run never reached, so (1, 10, 50, 100, 1000) works on a 200-cycle run
    edges = np.linspace(0, d.max(), bins + 1)                          # ONE set of bin edges shared by every curve --> if each histogram picked its own bins they would not be comparable
    for i in shown:
        ax.hist(d[:, i], bins=edges, histtype="step", linewidth=1.8, density=True, label=f"I = {i}")
        # histtype="step" draws outlines instead of filled bars, so four overlapping histograms stay readable
        # density=True normalizes each curve to area 1, so N = 10 and N = 1000 can be compared on the same axes without the bigger population simply being taller
    ax.set(xlabel="distance from source (um)", ylabel="probability density", title=f"Distance distribution, N = {result.n_cells}")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Step 35: Convergence, N Comparison, And The Control
# ---------------------------------------------------------------------------

# Mean distance versus iteration, ONE band per N, with the standard error shaded.
# Shows convergence as a continuous curve rather than as a handful of snapshots, and the shaded band visibly narrows as N grows.
def plot_convergence(results: Dict[int, SimulationResult], ax=None, nearest: bool = False):
    ax = _ax(ax)
    for n_cells in sorted(results):                        # sorted() so the legend reads 10, 100, 1000 rather than dictionary order
        trace = distance_trace(results[n_cells], nearest=nearest)
        x = np.arange(trace["mean"].size)
        line, = ax.plot(x, trace["mean"], linewidth=1.8, label=f"N = {n_cells}")   # the trailing comma unpacks the one-element list ax.plot returns
        ax.fill_between(x, trace["mean"] - trace["sem"], trace["mean"] + trace["sem"], alpha=0.22, color=line.get_color())
        # +/- one STANDARD ERROR, not one standard deviation --> this band is about how well we know the mean, and it is the band that shrinks like 1/sqrt(N)
    ax.set(xlabel="chemotaxis cycle I", ylabel="mean distance to source (um)", title="Convergence toward the source")
    ax.legend()
    return ax


# One ROW of histograms, one panel per N, all at the same iteration.
# Shows that larger N gives a smoother, tighter ESTIMATE of the same underlying distribution.
# THIS IS A DIFFERENT CLAIM from "larger I narrows the distribution" and the write-up has to keep the two apart --> here the true spread never changes, only how well we can see it.
def compare_N(results: Dict[int, SimulationResult], iteration: int = -1, bins: int = 30, nearest: bool = False):
    sizes = sorted(results)
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.8), sharex=True, sharey=True)   # shared axes so the panels are directly comparable by eye instead of each auto-scaling to itself
    axes = np.atleast_1d(axes)   # subplots() returns a bare axes object when there is only ONE panel, so wrap it to keep the loop below uniform
    for ax, n_cells in zip(axes, sizes):
        r = results[n_cells]
        d = (r.distances_to_nearest_source() if nearest else r.distances_to_source())[:, iteration]
        ax.hist(d, bins=bins, density=True, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(d.mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"mean {d.mean():.1f}")   # a vertical line at the mean, so the reader can see it lands in the same place for all three N
        ax.set(title=f"N = {n_cells}", xlabel="distance from source (um)")
        ax.legend()
    axes[0].set_ylabel("probability density")   # only the LEFTMOST panel needs a y label, since the axes are shared
    fig.suptitle(f"Distance distribution at I = {results[sizes[0]].n_iterations if iteration == -1 else iteration}")
    return fig, axes


# THE VALIDATION FIGURE: the biased population against the unbiased control.
# Without this panel, "the bacteria moved toward the source" is not yet evidence of anything --> the control had the same number of tumbles and the same run length, and used no information.
def plot_biased_vs_control(biased: SimulationResult, control: SimulationResult, ax=None):
    ax = _ax(ax)
    for label, r, color in (("biased", biased, "seagreen"), ("unbiased control", control, "gray")):
        trace = distance_trace(r)
        x = np.arange(trace["mean"].size)
        ax.plot(x, trace["mean"], linewidth=1.8, color=color, label=label)
        ax.fill_between(x, trace["mean"] - trace["sem"], trace["mean"] + trace["sem"], alpha=0.22, color=color)
    ax.set(xlabel="chemotaxis cycle I", ylabel="mean distance to source (um)", title="Biased run-and-tumble vs unbiased control")
    ax.legend()
    return ax
