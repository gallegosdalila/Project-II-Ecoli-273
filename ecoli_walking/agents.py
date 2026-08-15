# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# agents.py -- the movement layer.
# The MOVEMENT layer. This file knows how to move bacteria.
# It does not know what a concentration is, and it never decides which way to go --> simulate.py works out the direction and hands it in. This file only gets the geometry right.

# GROUP DECISION (Part 1): For speed we chose to use vectorized arrays rather than python objects.
                         # (we hold all N bacteria as ARRAYS (the Population class), not as N separate Python objects)

from typing import Optional

import numpy as np

from .config import Config


# ---------------------------------------------------------------------------
# Steps 12, 13, 14: Direction Helpers
# ---------------------------------------------------------------------------
# These live at module level (not inside Population) so the tests can call them directly with made-up inputs, instead of having to build a whole population first


# n uniformly random 2D unit vectors, shape (n, 2)
    # rng: the shared np.random.Generator passed down from main.py --> every random number in the project comes from this one object, which makes a run reproducible from one seed
    # n:   how many direction vectors we want
def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)                # theta --> one random heading ANGLE per bacterium, drawn uniformly on [0, 2pi), so no direction is preferred over any other
    return np.stack((np.cos(theta), np.sin(theta)), axis = -1)     # cos and sin of an angle always give a vector of length exactly 1, so we don't have to normalize afterwards. axis = -1 pairs them into (x, y)


# Random unit vectors with ANY leading shape, result (*shape, 2)
# This is the version tumble() needs: it draws all N bacteria x all 4 tumble steps in a single call, giving (N, 4, 2), instead of calling random_unit_vectors four times.
def random_unit_grid(rng: np.random.Generator, *shape: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=shape)            # size can be (N,) or (N, 4) or anything else --> numpy fills whatever shape we ask for
    return np.stack((np.cos(theta), np.sin(theta)), axis = -1)     # adds the trailing axis of size 2, so (N, 4) angles become (N, 4, 2) vectors


# Row-wise normalize: divide every vector by its own lengtth so it becomes length 1, leaving near-zero rows as ZERO instead of NaN
# A zero row means that this bacterium has no direction this cycle, which is a valid state in our model, not an error --> so it must not become NaN and mess-up everything downstream
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis = -1, keepdims = True)                    # length of each row. keepdims = True keeps the trailing axis so the shape is (N, 1) and the division broadcasts against (N, 2)
    return np.divide(v, norm, out = np.zeros_like(v), where = norm > eps)   # where= tells numpy to SKIP the rows that would divide by ~0, and out= says what those skipped rows should contain instead (zeros)


# ---------------------------------------------------------------------------
# Step 16: Domain Boundary
# ---------------------------------------------------------------------------

# Enforce the domain rule for every position at once. Vectorized and branch-free per element --> the `if`s below choose a RULE once, they do not loop over bacteria.
# GROUP DECISION: the default is "none"(unbounded). Reflect piles bacteria against the walls and wrap teleports them across the domain, and BOTH distort the distance-from-source histogram, which is our headline figure.
# Both are implemented and tested anyway, so the group can switch with one config change if we decide the physics needs it.
def apply_boundary(xy: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.boundary == "none":
        return xy                                          # nothing to do --> bacteria may swim as far as they like

    h = cfg.domain_half_width                              # h --> half the width of the box, so the domain runs from -h to +h
    span = 2.0 * h                                         # span --> the full width of the box

    if cfg.boundary == "wrap":
        return np.mod(xy + h, span) - h                    # shift so the domain starts at 0, take the remaidner (which folds anything past the edge back to the start), then shift back

    if cfg.boundary == "reflect":
        t = np.mod(xy + h, 2.0 * span)                     # t --> position along a TRIANGLE WAVE of period 2 * span, which folds the whole number line back into the box
        return np.where(t > span, 2.0 * span - t, t) - h   # once past the halfway point of the fold we count BACKWARDS instead of forwards, which is exactly what bouncing off a wall does

    raise ValueError(f"unknown boundary rule: {cfg.boundary!r}")   # fail with an error on a typo rather than silently doing nothing


# ---------------------------------------------------------------------------
# Step 17: Starting Positions
# ---------------------------------------------------------------------------

# Where the N bacteria begin, shape (N, 2).
# "uniform" scatters them across the whole domain, which is what makes the distance-from-source histogram START BROAD and visibly narrow as the iteration count grows
# Do NOT start every bacterium exactly on the source: distance would begin at 0 and could only grow, and the headline figure would say the opposite of what we want
def initial_positions(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    n = cfg.n_cells

    if cfg.start_mode == "point":
        return np.tile(np.asarray(cfg.start_point, dtype=float), (n, 1))   # tile repeats the ONE start point N times down the rows, giving (N, 2) --> every bacterium starts stacked at the same spot

    if cfg.start_mode == "uniform":
        h = cfg.domain_half_width
        return rng.uniform(-h, h, size=(n, 2))                             # every x and every y drawn independently between -h and +h

    if cfg.start_mode == "ring":
        return cfg.start_radius * random_unit_vectors(rng, n)              # unit vectors scaled by the radius put every bacterium the SAME distance out, just at random angles

    raise ValueError(f"unknown start_mode: {cfg.start_mode!r}")


# ---------------------------------------------------------------------------
# Step 18: The Vectorized Population
# ---------------------------------------------------------------------------

class Population:
    """All N bacteria held as arrays, not as N Python objects."""

    # What the object stores:
        # positions   (N, 2)  current x,y of every bacterium
        # directions  (N, 2)  unit vector of the most recent DIRECTED run --> kept so choose_run_direction has something to fall back on when the 4 tumbles cancel
        # state       (N,)    0 = tumbling, 1 = running. (useful for debugging sinngle cell) nothing reads it during run

    __slots__ = ("positions", "directions", "state", "cfg")   # __slots__ skips the per-object attribute dictionary --> saves memory, and turns a typo like pop.postions = ... into an error instead of a silent bug

    def __init__(self, positions: np.ndarray, cfg: Config, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.positions = np.asarray(positions, dtype=float).reshape(-1, 2)      # reshape(-1, 2) means "however many rows, 2 columns" --> accepts a list of pairs or an array and normalizes both to (N, 2)
        n = self.positions.shape[0]
        seed_rng = rng if rng is not None else np.random.default_rng(cfg.seed)  # a caller normally passes the ONE shared generator. If not, build one from the seed so a standalone call is still reproducible
        self.directions = random_unit_vectors(seed_rng, n)                      # every bacterium needs SOME initial heading, because the very first cycle may already need the fallback
        self.state = np.zeros(n, dtype=np.int8)                                 # using int8 not int64 --> since it holds 0 or 1, so (64 would be overkill))

    # Build a population straight from a Config, which is how run_simulation will do it later
    def from_config(cls, cfg: Config, rng: np.random.Generator) -> "Population":
        return cls(initial_positions(rng, cfg), cfg, rng)
    from_config = classmethod(from_config)   # classmethod() means it is called on the CLASS, as Population.from_config(cfg, rng). (Python fills in cls for us)

    def n(self) -> int:
        return self.positions.shape[0]
    n = property(n)   # property() means callers write pop.n, not pop.n()  since number of bacteria is a fact about the object, not an action it performs

    # -- movement primitives -------------------------------------------------

    # Take n_steps random reorientation steps for EVERY bacterium at once.
    # Returns the intermediate positions, shape (N, n_steps, 2), where [:, -1, :] is where the bacteria ended up. Also updates self.positions.
    # THIS IS THE np.cumsum TRICK: both the bacterium axis AND the step axis are vectorized, so the whole 4-step sequence comes from ONE cumulative sum. There is no Python loop over the sampling window either.
    def tumble(self, rng: np.random.Generator, n_steps: Optional[int] = None) -> np.ndarray:
        k = self.cfg.n_tumbles if n_steps is None else n_steps
        steps = self.cfg.tumble_step * random_unit_grid(rng, self.n, k)   # (N, k, 2) --> every bacterium's every step drawn in a single call to the generator
        path = self.positions[:, None, :] + np.cumsum(steps, axis=1)      # cumsum along the STEP axis turns individual steps into running positions: step1, step1+step2, step1+step2+step3, ...
                                                                          # [:, None, :] inserts an axis so the starting position (N, 2) broadcasts against the (N, k, 2) running total
        path = apply_boundary(path, self.cfg)
        self.positions = path[:, -1, :].copy()                            # [-1] along the step axis is the final position. .copy() so later writes to self.positions cannot silently edit the path we just returned
        self.state[:] = 0                                                 # mark every bacterium as tumbling (State 0 = tumbling)
        return path

    # Move every bacterium ONE directed step along its own unit vector.
    # `direction` is (N, 2) and is assumed already normalized, OR zero, which means "do not run this cycle".
    def run(self, direction: np.ndarray, length: Optional[float] = None) -> np.ndarray:
        step = self.cfg.run_length if length is None else length
        direction = np.asarray(direction, dtype=float).reshape(-1, 2)
        self.positions = apply_boundary(self.positions + step * direction, self.cfg)
        moving = np.linalg.norm(direction, axis=-1) > 0                            # moving = which bacteria actually ran this cycle (a zero vector means they did not)
        self.directions = np.where(moving[:, None], direction, self.directions)    # ONLY the movers update their remembered heading. The rest keep the old one, which is what the "prevoius fallback" will reuse next cycle
        self.state[:] = moving.astype(np.int8)                                     # 1 for the running bacterium, 0 for the ones that tumbled
        return self.positions
