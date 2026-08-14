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

# Only estimate_gradient and directional_derivative are written so far. Both take plain arrays and import nothing but numpy, which is why they can exist before agents.py or config.py do.
# estimate_gradient is the function most likely to be SUBTLY wrong: a sign error still runs and still produces a plausible-looking plot. That is why it is written this early, and why it gets four tests instead of one.
# Everything after this is either obviously right or obviously broken.

# Still to come, in this order:
#   choose_run_direction   --> needs Config for ascend and fallback
#   sense                  --> needs Config for sensing_noise
#   chemotaxis_cycle       --> needs Population from agents.py
#   run_simulation         --> needs Config and SimulationResult from contracts.py
#   run_unbiased_control   --> the baseline that makes run_simulation's result mean anything



from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Step 4: Gradient Estimation
# ---------------------------------------------------------------------------

# Estimate grad C from ONE concentration difference per cell. This is the MOST IMPORTANT FUNCTION in the project.
    # delta_c: How much concentration changed per bacterium          (N,)    C(t) - C(t - 4*dt)   (positive delta_c: concentration increased, negative: decreased)
    # displacement: How far each bacterium moved in x,yy directions  (N, 2)  x(t) - x(t - 4*dt)
# Returns g_est (N, 2), zeroed wherever the estimate is invalid, and valid (N,) bool, False where |d| was too small to divide by.
# Fully vectorized: the invalid rows are masked with np.where rather than skipped with an if, so the array shape never changes.
def estimate_gradient(delta_c: np.ndarray, displacement: np.ndarray, 
                      min_displacement: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:# min_displacement: the smallest movement considered safe for the calculation (1e-9)
                                                                                       # --> the function returns two NumPy arrays 
    d2 = np.einsum("ij,ij->i", displacement, displacement)    # squared distance traveled by each bacterium: d^2 = dx^2 + dy^2.   |d|^2 for every cell, shape (N,)  
    valid = d2 > min_displacement ** 2                        # compare squared quantities so we never take a square root just to test a threshold
    safe_d2 = np.where(valid, d2, 1.0)                        # if valid is True, keep its actual squraed distance. If valid is False, temporarily replace its squared distance with 1.0 (those rows get masked out below anyway)
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
    dist = np.linalg.norm(displacement, axis=-1)   # otal distance traveled by each bacterium using: distance = sqrt(dx^2 + dy^2)
    return np.divide(delta_c, dist, out=np.zeros_like(delta_c), where=dist > min_displacement)  # delta_c / dist: if dist is too small, return 0.0 instead of getting a divide-by-zero error 
