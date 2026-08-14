# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# simulate.py -- Section 3. Owner: Tracy.
# The biased random walk itself. This file owns the DECISION: given what the cell tasted over the last four time steps, which way does it run?

# THE ALGORITHM, one chemotaxis cycle per cell:
    # 1. Record x0 = the current position and C0 = C(x0).
    # 2. Take 4 random tumble steps. Call the endpoint x4, and C4 = C(x4).
    # 3. dC = C4 - C0 is exactly the "C(t) versus C(t - 4*dt)" comparison the assignment asks for, and d = x4 - x0 is the displacement those 4 tumbles produced.
    # 4. The directional derivative along d is dC / |d|, so the minimum-norm vector g satisfying g . d = dC is  g_est = (dC / |d|^2) * d
       # This is the best estimate of grad C available from a single directional sample, and on a linear field it equals the exact projection of the true gradient onto d.
    # 5. Run one directed step along +g_est for ascent. Because normalizing g_est just recovers sign(dC) * d_hat, the rule in plain language is:
       # "if it got better, keep going that way; if it got worse, turn around." That is what real E. coli do, and it is two lines of numpy.
    # 6. If |d| is essentially zero, the four tumbles cancelled and dC/|d|^2 would blow up, so fall back to some other direction.

# estimate_gradient is the function most likely to be SUBTLY wrong: a sign error still runs and still produces a plausible-looking plot. That is why it is written this early, and why it gets four tests instead of one.
# Everything after this is either obviously right or obviously broken.

# PHASE 5 completes this file: sense, chemotaxis_cycle, run_simulation and run_unbiased_control are all written below, so the algorithm now runs end to end.



from typing import Optional, Tuple

import numpy as np

from .agents import Population, normalize, random_unit_vectors   # Population is NEW in phase 5 --> chemotaxis_cycle drives it
from .config import Config                                       # choose_run_direction needs cfg.ascend and cfg.fallback
from .contracts import SimulationResult                          # NEW in phase 5 --> run_simulation fills one of these in and hands it to Sections 4 and 5
from .fields import ConcentrationField                           # NEW in phase 5 --> we finally need to actually SENSE the field, not just take deltas somebody else computed


# ---------------------------------------------------------------------------
# Step 4: Gradient Estimation
# ---------------------------------------------------------------------------

# Estimate grad C from ONE concentration difference per cell. This is the MOST IMPORTANT FUNCTION in the project.
    # delta_c: How much concentration changed per bacterium          (N,)    C(t) - C(t - 4*dt)   (positive delta_c: concentration increased, negative: decreased)
    # displacement: How far each bacterium moved in x,y directions   (N, 2)  x(t) - x(t - 4*dt)
# Returns g_est (N, 2), zeroed wherever the estimate is invalid, and valid (N,) bool, False where |d| was too small to divide by.
# Fully vectorized: the invalid rows are masked with np.where rather than skipped with an if, so the array shape never changes.
def estimate_gradient(delta_c: np.ndarray, displacement: np.ndarray,
                      min_displacement: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:# min_displacement: the smallest movement considered safe for the calculation (1e-9)
                                                                                       # --> the function returns two NumPy arrays
    d2 = np.einsum("ij,ij->i", displacement, displacement)    # squared distance traveled by each bacterium: d^2 = dx^2 + dy^2.   |d|^2 for every cell, shape (N,)
    valid = d2 > min_displacement ** 2                        # compare squared quantities so we never take a square root just to test a threshold
    safe_d2 = np.where(valid, d2, 1.0)                        # if valid is True, keep its actual squared distance. If valid is False, temporarily replace its squared distance with 1.0 (those rows get masked out below anyway)
    g_est = (delta_c / safe_d2)[:, None] * displacement       # g_est estimates a gradient vector for each bacterium. (delta_c / safe_d2) calculates one scaling value per bacterium.
                                                              # [:, None] changes array's shape from (N,) to (N, 1), allowing each scaling value to multiply both the x and y parts of the corresponding displacement.
    return np.where(valid[:, None], g_est, 0.0), valid        # returns 2 arrays: 1) Estimated Gradients: g_est when bacteriums movement was valid, [0.0, 0.0] when its movement was too small.
                                                                                # 2) Valid Boolean Mask: array indicating which rows are valid (True) and which are invalid (False)


# This function calculates how quickly the concentration changed per unit of distance traveled by each bacterium.
# Mathematically, it calculates: dC/ds ≈ ΔC / (distance traveled)
        # dC is the change in concentration
        # ds is the distance traveled
        # The output is one number per bacterium
# dC/ds along the sampled direction, shape (N,). Used by the tests, not by the simulation.

# delta_c: concentration change for each bacterium,
# displacement: the (x, y) movement of each bacterium.,
# min_displacement: the smallest distance considered safe for division.,
# -> np.ndarray: the function returns one NumPy array
def directional_derivative(delta_c: np.ndarray, displacement: np.ndarray, min_displacement: float = 1e-9) -> np.ndarray:
    dist = np.linalg.norm(displacement, axis=-1)   # total distance traveled by each bacterium using: distance = sqrt(dx^2 + dy^2)
    return np.divide(delta_c, dist, out=np.zeros_like(delta_c), where=dist > min_displacement)  # delta_c / dist: if dist is too small, return 0.0 instead of getting a divide-by-zero error


# ---------------------------------------------------------------------------
# Steps 5 and 6: Turning The Estimate Into A Run Direction
# ---------------------------------------------------------------------------

# Turn each bacterium's gradient ESTIMATE into the unit direction it will actually run, shape (N, 2).
    # g_est:    (N, 2)      the estimate that came out of estimate_gradient
    # valid:    (N,) bool   which of those estimates we're allowed to trust
    # previous: (N, 2)      each bacterium's most recent run direction, kept by Population --> this is what the "previous" fallback reuses
    # rng:      the shared generator, only touched if the fallback is "random"
    # cfg:      supplies ascend (which way is "better") and fallback (what to do with an untrustworthy estimate)
# Ascent uses +g_est, descent uses -g_est. Because normalizing g_est just recovers sign(dC) * d_hat. Prety much if it got better keep going, if it gets worse turn around
def choose_run_direction(g_est: np.ndarray, valid: np.ndarray, previous: np.ndarray,
                         rng: np.random.Generator, cfg: Config) -> np.ndarray:
    sign = 1.0 if cfg.ascend else -1.0                              # sign --> +1 climbs toward food, -1 swims away from it. One number flips the whole population's behaviour.
    unit = sign * normalize(g_est)                                  # normalize() turns each estimate into length 1 and leaves zero rows as ZERO rather than NaN, which is why the next line can test them safely
    usable = valid & (np.linalg.norm(unit, axis=-1) > 0)            # a row is usable only if the displacement was big enough (valid) AND the resulting vector is non-zero. & is elementwise "and" across the whole array.

    if cfg.fallback == "previous":
        backup = previous                                    # keep the bacterium's momentum --> it has no NEW information this cycle, so it shouldn'y throw away the old information either
    elif cfg.fallback == "random":
        backup = random_unit_vectors(rng, unit.shape[0])     # a fresh random heading, as if the cell simply tumbled again
    elif cfg.fallback == "skip":
        backup = np.zeros_like(unit)                         # a ZERO vector means "don't run this cycle" --> Population.run() checks for exactly this and leaves those bacteria where they are
    else:
        raise ValueError(f"unknown fallback: {cfg.fallback!r}")

    return np.where(usable[:, None], unit, backup)           # usable[:, None] is (N, 1), so the choice is made per bacterium and applies to BOTH its x and y components
    # In practice this fallback fires 0 times in 10^6 cycles, because four random unit steps essentially never cancel exactly. (The guard still has to exist, or the one time it happens we divide by ~0 and get NaN.)


# ---------------------------------------------------------------------------
# Step 20: Sensing The Field
# ---------------------------------------------------------------------------

# Read the concentration at a set of positions, optionally with noise on the reading.
    # field: any ConcentrationField (linear, single_source, competing_sources) --> we only ever call .concentration(), NEVER .gradient(), because a real cell can't perceive a gradient
    # xy:    (..., 2) positions
    # cfg:   supplies sensing_noise
# sensing_noise defaults to 0.0, so this is a PERFECT sensor unless the group turns it on. Real E. coli receptors are noisy, and Huo et al. showed that noise is what produces the Levy walk, so this is the hook for that experiment.
def sense(field: ConcentrationField, xy: np.ndarray, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    c = field.concentration(xy)                                     # ONE call handles all N bacteria at once, because of the (..., 2) shape contract in fields.py
    if cfg.sensing_noise > 0.0:
        c = c + rng.normal(0.0, cfg.sensing_noise, size=c.shape)    # add a normally-distributed error to every reading, drawn from the same shared generator so noisy runs are still reproducible
    return c


# Boolean mask of the bacteria sitting within capture_radius of ANY source, shape (N,). Only consulted when cfg.stop_at_source is True.
def _captured(xy: np.ndarray, field: ConcentrationField, cfg: Config) -> np.ndarray:
    src = np.atleast_2d(field.sources)
    dist = np.linalg.norm(xy[:, None, :] - src[None, :, :], axis=-1)   # pair every bacterium with every source --> (N, S)
    return dist.min(axis=-1) <= cfg.capture_radius                      # closest source only


# ---------------------------------------------------------------------------
# Step 21: One Full Chemotaxis Cycle
# ---------------------------------------------------------------------------

# Four tumbles, one gradient estimate, one directed run, for ALL N bacteria at once. This is steps 1 to 6 of the algorithm at the top of this file, in order.
# It FALLS OUT of the three pieces already written: estimate_gradient (step 10), Population.tumble/.run (step 18), and choose_run_direction (step 19). Nothing new is invented here, it's just assembly.
    # c_start: the concentration at the CURRENT position, passed in by run_simulation because it already knows it from the end of the previous cycle --> saves sampling the field twice for the same point
# Returns (new_positions, run_dirs, delta_c, valid, concentration_at_end, path) where path is (N, n_tumbles+2, 2): the start, each tumble, then the run endpoint. Section 5 uses path for trajectory plots.
def chemotaxis_cycle(pop: Population, field: ConcentrationField, rng: np.random.Generator,
                     cfg: Config, c_start: Optional[np.ndarray] = None):
    x0 = pop.positions.copy()                     # .copy() because pop.positions is about to be overwritten by the tumbles, and we need the ORIGINAL to compute the displacement
    if c_start is None:
        c_start = sense(field, x0, rng, cfg)      # normally run_simulation passes this in; this branch only runs if somebody calls chemotaxis_cycle on its own

    # STEPS 1 and 2: the sampling window. All 4 tumbles come from ONE cumsum inside Population.tumble, so there is no Python loop over the window.
    tumble_path = pop.tumble(rng)                 # (N, n_tumbles, 2)
    x4 = pop.positions
    c_end = sense(field, x4, rng, cfg)

    # STEP 3: the finite difference across the 4*dt window. THIS is the "C(t) versus C(t - 4*dt)" comparison the assignment asks for.
    delta_c = c_end - c_start
    displacement = x4 - x0

    # STEPS 4 to 6: decide, then move
    if cfg.biased:
        g_est, valid = estimate_gradient(delta_c, displacement, cfg.min_displacement)
        run_dir = choose_run_direction(g_est, valid, pop.directions, rng, cfg)
    else:
        run_dir = random_unit_vectors(rng, pop.n)          # UNBIASED CONTROL: identical number of tumbles and identical run length, but the direction uses no information at all
        valid = np.ones(pop.n, dtype=bool)                 # nothing was estimated, so nothing can be invalid

    if cfg.stop_at_source:
        captured = _captured(x4, field, cfg)
        run_dir = np.where(captured[:, None], 0.0, run_dir)   # a zero direction freezes the bacteria that have arrived, because Population.run() treats zero as "don't move"

    pop.run(run_dir)
    c_final = sense(field, pop.positions, rng, cfg)

    path = np.concatenate((x0[:, None, :], tumble_path, pop.positions[:, None, :]), axis=1)   # glue start + 4 tumbles + run endpoint into one (N, n_tumbles+2, 2) block
    return pop.positions.copy(), run_dir, delta_c, valid, c_final, path


# ---------------------------------------------------------------------------
# Step 23: The Driver
# ---------------------------------------------------------------------------

# Run cfg.n_cells bacteria for cfg.n_iterations chemotaxis cycles and return a filled SimulationResult.
# GROUP DECISION (Part 1, "How random numbers are handled"): ONE generator is created in main.py and threaded down through every call.
    # Passing rng=None here builds one from cfg.seed, so a standalone call is still reproducible, but production code should pass its own.
    # Module-level np.random cannot be seeded reliably, which is why we never use it anywhere in this package.
# Every output array is PREALLOCATED with np.empty and filled by index --> no .append, and no list growing inside the loop.
def run_simulation(field: ConcentrationField, cfg: Config,
                   rng: Optional[np.random.Generator] = None) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    n, iters, k = cfg.n_cells, cfg.n_iterations, cfg.n_tumbles

    pop = Population.from_config(cfg, rng)

    positions = np.empty((n, iters + 1, 2))       # +1 because index 0 holds the STARTING position, before any cycle has run
    concentrations = np.empty((n, iters + 1))
    run_directions = np.empty((n, iters, 2))
    delta_c = np.empty((n, iters))
    gradient_valid = np.empty((n, iters), dtype=bool)
    substeps = np.empty((n, iters, k + 2, 2)) if cfg.store_substeps else None   # only allocated when asked for --> at N=1000 and I=1000 this array alone is about 80 MB

    positions[:, 0, :] = pop.positions
    concentrations[:, 0] = sense(field, pop.positions, rng, cfg)

    # THE ONE UNAVOIDABLE PYTHON LOOP: cycle i+1 depends on where cycle i left the bacteria, so it cannot be vectorized away.
    # But it costs only O(iters) passes, and each pass does O(N) work inside numpy --> 1000 passes over 1000 bacteria, not 1,000,000 separate operations.
    for i in range(iters):
        xy, u, dc, valid, c, path = chemotaxis_cycle(pop, field, rng, cfg, c_start=concentrations[:, i])   # last cycle's ENDING concentration is this cycle's STARTING one, so the field is never sampled twice for the same point
        positions[:, i + 1, :] = xy
        concentrations[:, i + 1] = c
        run_directions[:, i, :] = u
        delta_c[:, i] = dc
        gradient_valid[:, i] = valid
        if substeps is not None:
            substeps[:, i] = path

    result = SimulationResult(
        positions=positions,
        concentrations=concentrations,
        run_directions=run_directions,
        delta_c=delta_c,
        gradient_valid=gradient_valid,
        field_name=field.name,
        sources=np.atleast_2d(field.sources),   # COPY the source locations in, so plotting and stats never need the field object itself
        config=cfg.as_dict(),                   # record the exact settings alongside the data, so we can never wonder which parameters produced a figure
        substeps=substeps,
    )
    result.validate()   # fail HERE, at the handoff, rather than inside somebody else's figure
    return result


# ---------------------------------------------------------------------------
# Step 25: The Unbiased Control
# ---------------------------------------------------------------------------

# Same bacteria, same number of tumbles, same run length --> but the run direction is chosen at RANDOM instead of from the gradient estimate
    # Until the control exists, "the bacteria moved toward the source" is UNFALSIFIABLE meaning a bug that walked every cell toward the origin no matter what the feild looked like would pass every test written so far (FIXED HERE)
    # test_biased_beats_unbiased_control is the first test that could actually catch that, because the control has the identical bug and would move identically.
# replace() returns a NEW Config with biased=False and leaves the original untouched, which is why cfg can be safely reused for the biased run.
def run_unbiased_control(field: ConcentrationField, cfg: Config,
                         rng: Optional[np.random.Generator] = None) -> SimulationResult:
    from dataclasses import replace
    return run_simulation(field, replace(cfg, biased=False), rng)
