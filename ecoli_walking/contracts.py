# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# contracts.py -- the ONE data structure that crosses section boundaries.
    # Section 3 (simulate.py) FILLS a SimulationResult in
    # Section 4 (experiments.py, stats.py) SAVES it and computes statistics on it
    # Section 5 (plotting.py) READS it to draw the figures

# GROUP DECISION (Part 1, "What is in a SimulationResult"): ARRAYS.
    # We chose positions shaped N x T x 2 and concentrations shaped N x T, over "a list of Bacterium objects", because stats.py needs to do vectorized numpy math over the whole population at once.
    # A list of objects would force a Python loop in every single statistic we compute.


from dataclasses import dataclass, field   # dataclass writes __init__ for us from the annotations below, so we don't hand-write boilerplate
from typing import Optional, Dict, Any     # type hints only, no runtime effect
import numpy as np                         # positions and concentrations are numpy arrays, never python lists


class SimulationResult:
    """Everything one simulation run produced."""

    # ARRAY SHAPES, where N = number of bacteria, I = number of chemotaxis cycles, and the trailing 2 is always (x, y):
    positions: np.ndarray        # (N, I+1, 2) --> position at the END of each cycle. Index 0 along the middle axis is the STARTING position, which is why it's I+1 and not I.
    concentrations: np.ndarray   # (N, I+1)    --> the concentration each bacterium sensed at each of those positions
    run_directions: np.ndarray   # (N, I, 2)   --> the unit vector each bacterium ran along in each cycle. Useful for checking the bias points the right way.
    delta_c: np.ndarray          # (N, I)      --> C(t) minus C(t - 4*dt) for each cycle. This is the finite difference the whole algorithm rests on, so it's worth keeping.
    gradient_valid: np.ndarray   # (N, I) bool --> False wherever the 4 tumbles cancelled and the gradient estimate had to fall back
    field_name: str              # which concentration profile was used: linear, shallow, single_source, or competing_sources
    sources: np.ndarray          # (S, 2)      --> where the peaks are, COPIED from the field so plotting and stats never need the field object itself
    config: Dict[str, Any] = field(default_factory=dict)   # the exact Config used, as a plain dict --> a saved run records its own settings, so we can never wonder which parameters produced a figure
    substeps: Optional[np.ndarray] = None                  # (N, I, n_tumbles+2, 2) the full within-cycle path, or None --> only stored when config['store_substeps'] is True

    # Every array above is PREALLOCATED by run_simulation with np.empty and filled by index. There is no .append anywhere in this package.
    # default_factory=dict is needed because a plain {} as a default would be SHARED by every SimulationResult ever made --> the factory builds a fresh empty dict each time.


    # ---- shape helpers (so callers never have to remember which axis is which) ----

    def n_cells(self) -> int:
        return self.positions.shape[0]      # axis 0 is the bacterium axis
    n_cells = property(n_cells)             # property() means callers write result.n_cells, not result.n_cells()

    def n_iterations(self) -> int:
        return self.positions.shape[1] - 1  # minus 1 because index 0 is the starting position, not a completed cycle
    n_iterations = property(n_iterations)


    # ---- STEP 24: the derived quantities Section 4 and Section 5 both need ----

    # Straight-line distance from every bacterium to ONE source, at EVERY iteration. Returns shape (N, I+1).
    # THIS IS THE HEADLINE MEASUREMENT of the whole project: the assignment asks how the distance-from-source distribution narrows as N and I increase, and this is the array that answers it.
    def distances_to_source(self, source_index: int = 0) -> np.ndarray:
        src = np.atleast_2d(self.sources)[source_index]        # atleast_2d guards against a field that returned a bare (2,) instead of (S, 2), so this works either way
        return np.linalg.norm(self.positions - src, axis=-1)   # subtract the source from every position (broadcasting), then norm along the LAST axis to collapse each (dx, dy) pair into one distance

    # Distance to whichever source happens to be CLOSEST, shape (N, I+1).
    # Needed for competing_sources, where "distance from THE source" is ambiguous --> a bacterium that found the second peak has not failed, it just picked the other basin.
    def distances_to_nearest_source(self) -> np.ndarray:
        src = np.atleast_2d(self.sources)                                    # (S, 2)
        diff = self.positions[:, :, None, :] - src[None, None, :, :]         # the None entries insert axes so EVERY position is paired with EVERY source --> (N, I+1, S, 2), no loop
        return np.linalg.norm(diff, axis=-1).min(axis=-1)                    # distance to each source, then keep the smallest one per bacterium per iteration

    # Vector from start to finish for each bacterium, shape (N, 2). Averaged over the population this shows the NET DRIFT direction, which is what proves the walk is biased on a linear field.
    def net_displacement(self) -> np.ndarray:
        return self.positions[:, -1, :] - self.positions[:, 0, :]

    # Check the shapes agree BEFORE handing this to another section, so a mistake surfaces at the handoff instead of inside somebody else's figure two hours later.
    def validate(self) -> None:
        n, t_plus_1, d = self.positions.shape
        assert d == 2, f"positions must be 2D, got last dim {d}"
        assert self.concentrations.shape == (n, t_plus_1), f"concentrations {self.concentrations.shape} != expected {(n, t_plus_1)}"
        assert self.run_directions.shape == (n, t_plus_1 - 1, 2)   # one FEWER than positions, because index 0 of positions is the start, not a cycle
        assert self.delta_c.shape == (n, t_plus_1 - 1)
        assert self.gradient_valid.shape == (n, t_plus_1 - 1)
        assert np.atleast_2d(self.sources).shape[1] == 2
        assert np.isfinite(self.positions).all(), "non-finite positions --> something divided by zero or diverged"   # catches NaN and inf, which numpy would otherwise carry silently into every plot

SimulationResult = dataclass(SimulationResult)   # dataclass() writes __init__ from the annotations above, so we never hand-write "self.positions = positions" nine times
