# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# stats.py -- statistics computed on a finished run
# Turns a SimulationResult into the NUMBERS THAT GO IN THE REPORT: 
    # distance from source, and the mean / variance / std / median of that distance at whichever iteration counts (I = 1, 10, 50, 100, 1000...)

# Everything here is a PURE FUNCTION on arrays:
    # nothing in this file runs a simulation. This filer it ONLY READS a SimulationResult somebody else already produced
    # also, nothing in this file draws a figure (plotting.py will do that, reading these same numbers)
# All vectorized numpy plus comprehensions. No Python for-loops over individual bacteria

# now that run_simulation is complete, we can now calculate statistics. run_simulatiom produces actual results that we can compute stats on.

# What stats.py will compute: 
    # mean, median, std, variance, min, max, q25 and q75 of the distance distribution at selected iteration counts, 
    # squared displacement (MSD), and the chemotactic drift (mean inward speed toward the source)

from typing import Dict, Iterable

import numpy as np

from .contracts import SimulationResult


# ---------------------------------------------------------------------------
# Step 27: Summary of Bacterial Distances at Single Iteration
# ---------------------------------------------------------------------------


# The function takes the 1D arrat distances of all N bacteria from a source at one moment and calculates a statistical summary: 
    # count, average, median, spread, minimum, maximum, and quartiles
# Returns a plain dict so it drops straight into a printed table or a pandas DataFrame without conversion.
def distance_stats(distances: np.ndarray) -> Dict[str, float]:
    d = np.asarray(distances, dtype=float).ravel()   # ravel() flattens whatever shape came in, so a caller can pass a row or a column without us caring which. SOURCE: https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
    return {
        "n": d.size,
        "mean": float(np.mean(d)),                                    # the average distance from the source across the whole population
        "median": float(np.median(d)),                                # less sensitive than the mean to a few stragglers that never found the source --> report BOTH, since a big gap between them means the distribution is skewed
        "std": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,       # measure how widely bacterial distances vary using sample SD (N−1) (ddof=1); report 0.0 when only one distance exists
        "variance": float(np.var(d, ddof=1)) if d.size > 1 else 0.0,  # measure the spread of bacterial distances as squared deviations from the mean using sample variance (N−1); report 0.0 when only one distance exists
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "q25": float(np.percentile(d, 25)),                           # first quartile: the distance at or below which approximately 25% of bacteria fall; describes the lower part of the distribution without assuming symmetry
        "q75": float(np.percentile(d, 75)),
    }



# ---------------------------------------------------------------------------
# Step 28: How Bacterial Distances Change Over the Entire Simulation
# ---------------------------------------------------------------------------

# Summarize bacterial distances at every saved simulation time point
# Mean, median, std and standard error of the distance at EVERY iteration (allows plotting.py to plot convergence curves without recomputing anything)
# Each returned array has length I+1 (including the initial state), matching the middle axis of result.positions
    # nearest: True measures distance to whichever source is CLOSEST. Only matters for competing_sources, where "the" source is ambiguous
def distance_trace(result: SimulationResult, nearest: bool = False) -> Dict[str, np.ndarray]:
    d = result.distances_to_nearest_source() if nearest else result.distances_to_source()   # (N, I+1)
    return {
        "mean": d.mean(axis=0),                                                            # axis=0 averages over BACTERIA, leaving one value per iteration
        "median": np.median(d, axis=0),
        "std": d.std(axis=0, ddof=1) if result.n_cells > 1 else np.zeros(d.shape[1]),      # std measures how widely individual bacterial distances vary
        "sem": (d.std(axis=0, ddof=1) / np.sqrt(result.n_cells)) if result.n_cells > 1 else np.zeros(d.shape[1]),   # sem estimates the uncertainty in the mean distance: std / sqrt(N)
    }
# A decreasing mean indicates movement toward the source, while a decreasing std indicates that bacterial distances are becoming more similar.

    # THE TWO ARE DIFFERENT CLAIMS AND THE WRITE-UP MUST KEEP THEM APART:
        # std falling  = larger I --> the bacteria really are closer together
        # sem falling  = larger N --> our ESTIMATE of where they are is sharper, even though the true spread never changed
    # This is exactly the "how the distance distribution narrows as N and iteration count I increase" the assignment asks about, and it narrows for two unrelated reasons.



# ---------------------------------------------------------------------------
# Step 29: Additional Population Statistics and Movement Metrics
# ---------------------------------------------------------------------------

# distance_stats evaluated at each requested iteration index, returned as a dict keyed by iteration count.
# Iterations past the end of the run are silently SKIPPED, so we can pass (1, 10, 50, 100, 1000) to a 200-cycle run and get back only the four that exist, instead of an IndexError.
def snapshot_stats(result: SimulationResult, iterations: Iterable[int], nearest: bool = False) -> Dict[int, Dict[str, float]]:
    d = result.distances_to_nearest_source() if nearest else result.distances_to_source()
    return {int(i): distance_stats(d[:, i]) for i in iterations if 0 <= i <= result.n_iterations}


# Compute mean squared displacement using the standard MSD definition; 
# this analysis was included for a qualitative comparison with Huo et al. Figure 2, but the Python implementation is specific to our simulation data

# Ensemble-averaged MEAN SQUARED DISPLACEMENT versus lag time, for comparison with Huo et al. Fig 2.
    # They fit MSD = D * t^alpha and measured alpha = 1.66 for wild-type cells (SUPERdiffusive, spreading faster than a normal random walk) and alpha = 1.09 for the mutant with no signalling noise (normal diffusion)
    # Huo et al. use a similar MSD analysis in Figure 2, but our comparison is qualitative because our model uses 2D cycle endpoints and does not explicitly simulate CheY-P dynamics.
    # Our model has NO CheY-P signalling noise, so in a homogeneous field we would expect alpha near 1
# "Lag" is measured in chemotaxis cycles rather than seconds because result.positions stores each bacterium's position at the end of every cycle.

# Calculate mean squared displacement (MSD) at every possible lag by averaging squared movement over all bacteria and all valid starting times.
# MSD can later be fit to MSD proportional to lag^alpha, where alpha near 1 indicates normal diffusion and alpha greater than 1 indicates superdiffusion.
# Huo et al. use a similar MSD analysis in Figure 2, but our comparison is qualitative because our model uses 2D cycle endpoints and does not explicitly simulate CheY-P dynamics.
# Lag is measured in chemotaxis cycles rather than seconds because result.positions stores each bacterium's position at the end of every cycle.
def mean_squared_displacement(result: SimulationResult) -> np.ndarray:
    pos = result.positions                    # (N, T, 2)
    n_lags = pos.shape[1]
    lags = np.arange(1, n_lags)
    # For each lag, compare EVERY pair of positions that far apart in time, across every bacterium at once. The comprehension is over lags only, never over bacteria.
    msd = np.array([np.mean(np.sum((pos[:, lag:, :] - pos[:, :-lag, :]) ** 2, axis=-1)) for lag in lags])
    return msd


# Mean inward speed toward the source, in um per cycle. POSITIVE means the population is closing on the source.
# This is the single number that most cleanly separates the biased run from the unbiased control, so it is the one to put in the results table next to each condition.
def chemotactic_drift(result: SimulationResult) -> float:
    d = result.distances_to_source()
    return float((d[:, 0].mean() - d[:, -1].mean()) / result.n_iterations)   # total distance closed, divided by how many cycles it took


# Share of the population sitting inside `radius` of the source at a given iteration. Defaults to the FINAL iteration.
def fraction_within(result: SimulationResult, radius: float, iteration: int = -1) -> float:
    d = result.distances_to_source()[:, iteration]
    return float(np.mean(d <= radius))   # d <= radius gives an array of True/False, and the mean of a boolean array IS the fraction that are True --> no counting loop needed