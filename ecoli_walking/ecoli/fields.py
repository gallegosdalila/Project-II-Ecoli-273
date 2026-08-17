# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy


# This file defines the chemical environment that bacteria swim through.
# Purpose: At position (x, y), how much attractant is there? (the function C(x, y)).
# 3 Versions: 1) Shallow ramp, 2) One food source, 3) Two competing food sources.
#   Each of the 3 versions is a subclass of ConcentrationField.

# Two methods per field, used by different files in this project:
#   1) concentration(xy):
        # What a cell can smell(the only thing the cell is allowed to sense).
        # simulate.py (the biased random walk) calls this to get the concentration at the cell's current position.
#   2) gradient(xy):
        # The uphill direction (towards higher concentration of food source), from the following equation:.
                # grad C = ( dC/dx , dC/dy )
            # Each field has its own, since each has its own C:
                # LinearGradient    grad C = g                        (a plane: same slope everywhere)
                # GaussianSource    grad C = -C (x - x_s) / sigma^2   (chain rule on the exponential)
                # CompetingSources  the per-source Gaussian gradients, added up
        # TESTS ONLY, never call this from simulate.py(the biased random walk) since an cell cannot perceive the gradient;
        # it has to estimate it from how concentration changed over the last 4 tumbles.
        # gradient(xy) is the answer key we grade estimate_gradient() (the cell's guess) against.


from abc import ABC, abstractmethod    # lets us force subclasses to fill in methods
from typing import Sequence, Tuple     # type hints only, no runtime effect
import numpy as np                     # for vectorization

# Will allow simulate.py to accept ANY field without knowing which one it got.
# ABC blocks instantiation until every abstract method is filled in, (a half-finished field can't reach simulation)
class ConcentrationField(ABC):
    """Interface every concentration field obeys."""

    name = "field"      # short ID for filenames and plot titles; subclasses override

    # The only method the simualted bacterium may call.
    def concentration(self, xy: np.ndarray) -> np.ndarray: ...
    concentration = abstractmethod(concentration)

    # The exact gradient of C from the equation measures which way and speed the concentration rises
    # equation = the pair of partial derivatives:    grad C = ( dC/dx , dC/dy )
    # There is no one formula here in the base class -- each subclass differentiates its own C
    # In simulate.py, DO NOT CALL gradient() since the cell cannot perceive the gradient; 
    # it has to estimate it from how concentration changed over the last 4 tumbles.
    # Only ours test will call it to check estimate_gradient() against the right answer
    def gradient(self, xy: np.ndarray) -> np.ndarray: ...
    gradient = abstractmethod(gradient)


    # Where the peaks are. stats.py measures "distance from source" against this, so every field has to name its own target(s)
    def sources(self) -> np.ndarray:
        return np.zeros((1, 2))    # (num of sources(S) == 1, 2D position x & y)
    sources = property(sources)    # make sources readable. property() means callers write f.sources, not f.sources() 
                                   # with () f.sources() would get the function itslef, not the answer)

 
    # -- helpers the base class provides; subclasses inherit these -------------------------------------------------
    

    # Every public method starts by calling this, so bad input fails here with a clear message
    # instead of as a broadcasting error deep in the math.
    def _as_points(xy) -> np.ndarray:
        arr = np.asarray(xy, dtype=float)   # accept lists/ints; force float
        if arr.shape[-1] != 2:              # last axis must be (x, y)
            raise ValueError(f"expected last axis of size 2, got {arr.shape}")
        return arr
    _as_points = staticmethod(_as_points)   # staticmethod() = no self needed. Leading _ marks it internal

    # The measuring instrument: dC/dx ~ [C(x+h,y) - C(x-h,y)] / 2h
    # Deliberately dumb and obviously correct, so it can be trusted to check the hand-derived calculus in each subclass. Tests only.
    # "Central" means we step BOTH ways and split the difference, which cancels the leading error term and beats a one-sided step.
    def numerical_gradient(self, xy, h: float = 1e-5) -> np.ndarray:
        p = self._as_points(xy)                     # validate and convert the input
        ex = np.zeros_like(p)                       # an all-zeros array shaped like p
        ex[..., 0] = h                              # put h in the x slot only, so ex is a tiny nudge along x
        ey = np.zeros_like(p)                       # same again for y
        ey[..., 1] = h                              # ey is a tiny nudge along y
        dx = (self.concentration(p + ex) - self.concentration(p - ex)) / (2 * h)   # one central difference in x, evaluated for ALL points at once
        dy = (self.concentration(p + ey) - self.concentration(p - ey)) / (2 * h)   # same in y
        return np.stack((dx, dy), axis=-1)          # glue the two into (..., 2) so the shape matches what gradient() returns

    # (X, Y, C) grids for plotting.py's contour plots. Lives here, not in plotting.py, so the figure and the simulation read the same C(x, y).
    def meshgrid(self, half_width: float = 100.0, n: int = 200
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        axis = np.linspace(-half_width, half_width, n)      # n evenly spaced values from -H to +H
        X, Y = np.meshgrid(axis, axis)                      # two (n, n) grids holding every x and every y
        C = self.concentration(np.stack((X, Y), axis=-1))   # stack into (n, n, 2) so the SAME concentration() handles the grid and returns (n, n)
        return X, Y, C                                      # exactly the three arrays plt.contourf wants


class LinearGradient(ConcentrationField):
    """Shallow ramp:  C = c0 + g . (x - x_ref)"""

    # The validation field, and the anchor for the whole test suite: its gradient is the same vector everywhere, so the right answer can be checked on paper.
    # That is why it is written before any curved field. Doubles as the "shallow gradient" profile required by the spec if the slope is small.

    name = "linear"

    # Arguments with defaults, not hardcoded numbers, so the ramp can be changed without editing this file.
    def __init__(self, slope: Sequence[float] = (0.01, 0.0),
                 c0: float = 1.0, reference: Sequence[float] = (0.0, 0.0)):
        self.slope = np.asarray(slope, dtype=float).reshape(2)           # slope --> g, the constant gradient vector; reshape(2) errors loudly if someone passes 3 numbers
        self.c0 = float(c0)                                              # c0 --> the concentration AT the reference point
        self.reference = np.asarray(reference, dtype=float).reshape(2)   # reference --> x_ref, the one point where C is exactly c0

    # C = c0 + g.(x - x_ref) at every input point.
    # Hand checked: slope=(2,3), c0=5, ref=(1,1) -> C(2,1) must be 7.0
    def concentration(self, xy):
        p = self._as_points(xy)                              # validate shape first
        return self.c0 + (p - self.reference) @ self.slope   # @ is the dot product; it contracts the last axis, so (..., 2) @ (2,) gives (...) for every point at once

    # A plane has the same slope everywhere, so nothing is computed (just repeat `slope` to match input shape)
    def gradient(self, xy):
        p = self._as_points(xy)                              # only needed for its SHAPE, the values are irrelevant here
        return np.broadcast_to(self.slope, p.shape).copy()   # broadcast_to makes a cheap read-only view; .copy() turns it into a normal writable array callers can modify

    # A plane has no true maximum, but stats.py measures distance-from-source for every field. Report a point far uphill so that metric still works.
    def sources(self):
        direction = self.slope / (np.linalg.norm(self.slope) or 1.0)   # `or 1.0` guards a completely flat field: norm would be 0.0, which is false, so we divide by 1 instead of producing NaN
        return (100.0 * direction).reshape(1, 2)                       # reshape to (1, 2): a list holding one point, so callers never special-case the count
    sources = property(sources)


class GaussianSource(ConcentrationField):
    """Single source:  C = A exp(-|x - x_s|^2 / (2 sigma^2))"""

    # A food pellet dissolving in still water: concentration is highest at the source and falls off smoothly with distance. This is the main field for the headline result.
    # The Gaussian is not arbitrary; it's the solution to the 2D diffusion equation for a point release, with sigma^2 = 2Dt, so the smell spreads as sqrt(t).
    # Differentiating by the chain rule gives  grad C = -C (x - x_s) / sigma^2
    # (x - x_s) points AWAY from the source and the minus sign flips it, so grad C points INWARD. That is why gradient ASCENT walks cells toward food.

    name = "single_source"

    def __init__(self, center: Sequence[float] = (0.0, 0.0),
                 amplitude: float = 100.0, sigma: float = 50.0,
                 baseline: float = 0.0):
        self.center = np.asarray(center, dtype=float).reshape(2)   # center --> x_s, where the food is
        self.amplitude = float(amplitude)                          # amplitude --> A, the peak concentration right at the source
        self.sigma = float(sigma)                                  # sigma --> width; how far the smell carries. Too small and distant cells sense nothing at all.
        self.baseline = float(baseline)                            # baseline --> background concentration far away from any source

    # Check by confirming C(center) == amplitude + baseline.
    def concentration(self, xy):
        p = self._as_points(xy)                                            # validate the shape first
        d = p - self.center                                                # d --> (..., 2) offset from the source to each point
        r2 = np.einsum("...i,...i->...", d, d)                             # |d|^2 summed over the last axis; einsum fuses the multiply and the sum in one pass, and works for (2,), (N,2) and (200,200,2) alike
        return self.baseline + self.amplitude * np.exp(-r2 / (2.0 * self.sigma ** 2))   # the Gaussian itself

    # -C (x - x_s) / sigma^2. First real calculus in the file, and what numerical_gradient exists to verify.
    def gradient(self, xy):
        p = self._as_points(xy)
        d = p - self.center                                                # same offset as above
        r2 = np.einsum("...i,...i->...", d, d)                             # same |d|^2, computed inline rather than by calling concentration() again, which would redo all of this work
        bump = self.amplitude * np.exp(-r2 / (2.0 * self.sigma ** 2))      # C minus the baseline; a constant baseline contributes nothing to a derivative
        return -(bump / self.sigma ** 2)[..., None] * d                    # [..., None] adds a trailing axis so the scalar per point broadcasts against d's (x, y) pair

    # The peak is exactly the center; reshape to (1, 2).
    def sources(self):
        return self.center.reshape(1, 2)
    sources = property(sources)


class CompetingSources(ConcentrationField):
    """Sum of several Gaussians with different strengths."""

    # The population should SPLIT, with the saddle point between the peaks acting as a watershed.
    # Cells starting on either side climb into different basins, so keep the amplitudes unequal and one basin visibly wins.
    # VALIDATED THIS by confirming it equals the sum of the GaussianSources. (LinearGradient and GaussianSource are checked against numerical_gradient; 
    # CompetingSources is checked against the sum of parts instead, because there's no independent right answer for it.)

    name = "competing_sources"

    def __init__(self,
                 centers: Sequence[Sequence[float]] = ((-50.0, 0.0), (50.0, 0.0)),
                 amplitudes: Sequence[float] = (100.0, 60.0),
                 sigmas: Sequence[float] = (30.0, 30.0),
                 baseline: float = 0.0):
        self.centers = np.asarray(centers, dtype=float).reshape(-1, 2)      # (S, 2); reshape(-1, 2) means "however many rows, 2 columns"
        self.amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)   # (S,) one peak height per source
        self.sigmas = np.asarray(sigmas, dtype=float).reshape(-1)           # (S,) one width per source
        self.baseline = float(baseline)
        # Catch a mismatched setup HERE, at construction, rather than as a confusing broadcasting error later inside a 1000-cell run.
        if not (len(self.centers) == len(self.amplitudes) == len(self.sigmas)):
            raise ValueError("centers, amplitudes and sigmas must be same length")

    # The per-source pieces both concentration() and gradient() need, factored out so the exponential is computed once and the two formulas cannot drift apart.
        # returns  d    (..., S, 2)  displacement from each source
                # bump (..., S)     each source's Gaussian value
    def _bumps(self, p: np.ndarray):
        d = p[..., None, :] - self.centers                                # [None] inserts an axis making p (..., 1, 2), which broadcasts against (S, 2) to pair every point with every source, no loop
        r2 = np.einsum("...i,...i->...", d, d)                            # (..., S) squared distance to each source
        bump = self.amplitudes * np.exp(-r2 / (2.0 * self.sigmas ** 2))   # the (S,) parameters broadcast across the source axis
        return d, bump

    # Concentrations from independent sources simply add. Sum bump over the S axis.
    def concentration(self, xy):
        p = self._as_points(xy)
        _, bump = self._bumps(p)                    # the underscore means we are deliberately ignoring d here
        return self.baseline + bump.sum(axis=-1)    # axis=-1 is the source axis

    # Gradients add too, because differentiation is linear. Apply the GaussianSource formula per source, then sum over the S axis.
    def gradient(self, xy):
        p = self._as_points(xy)
        d, bump = self._bumps(p)                                     # here we need both pieces
        per_source = -(bump / self.sigmas ** 2)[..., None] * d       # (..., S, 2) each source's own gradient contribution
        return per_source.sum(axis=-2)                               # axis=-2 is the source axis; axis=-1 is x/y, which we must keep

    # Already stored as (S, 2); return as-is.
    def sources(self):
        return self.centers
    sources = property(sources)


# =========================================================================
# [8] Registry -- lets main.py pick a field by name from the command line
# =========================================================================
# Values are lambdas, not instances, so each make_field() call returns a FRESH object. Sharing one instance across a sweep would let a mutation in one run leak into the next.

FIELDS = {
    "linear": lambda: LinearGradient(slope=(0.02, 0.01)),
    "shallow": lambda: LinearGradient(slope=(0.001, 0.0005)),   # same class, gentler slope -- this is the "shallow gradient" profile the spec requires
    "single_source": lambda: GaussianSource(),
    "competing_sources": lambda: CompetingSources(),
}


# The only way main.py creates a field. On a bad name, raise with the list of valid options rather than a bare KeyError that tells the user nothing.
def make_field(name: str) -> ConcentrationField:
    if name not in FIELDS:
        raise KeyError(f"unknown field {name!r}; choose from {sorted(FIELDS)}")
    return FIELDS[name]()   # call the lambda, which builds a new instance


