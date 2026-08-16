# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# experiments.py -- population sweeps and saved runs.
# Sweeps over N and over seeds, plus the save-and-load layer, so that plotting.py NEVER has to rerun a simulation just to redraw a figure
# Define the populaiton sizes N = 10, 100, 1000 and the iteration checkpoints at which population positions will be saved

# WHY SAVING MATTERS, and not just for speed:
    # If plotting.py could rerun the simulation, then a figure and the results table could quietly come from two DIFFERENT runs and nobody would notice.
    # Saving once and reading the same file everywhere makes that impossible. It is a correctness guarantee first and a convenience second.

from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .config import Config, POPULATION_SIZES
from .contracts import SimulationResult
from .fields import make_field
from .simulate import run_simulation


# ONE naming convention for saved runs, so figures.py can find any file BY RULE instead of by guessing or hardcoding a list
# Everything that distinguishes one run from another goes in the filename: which field, biased or control, how many bacteria, which seed
# Example: single_source_biased_N1000_seed0.npz
def result_path(out_dir, field_name: str, n_cells: int, seed: int, biased: bool = True) -> Path:
    tag = "biased" if biased else "control"
    return Path(out_dir) / f"{field_name}_{tag}_N{n_cells}_seed{seed}.npz"


class ExperimentRunner:
    """Runs the N = 10, 100, 1000 sweep and writes the results to disk."""

    def __init__(self, field_name: str = "single_source", base_config: Config = None, out_dir="results"):
        self.field_name = field_name           # which concentration profile every run in this sweep uses
        self.base = base_config or Config()    # the settings shared by every run (only n_cells, seed and biased vary on top of it)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)   # parents=True makes any missing folders above it too; exist_ok=True means rerunning is not an error

    # ONE simulation, at one population size and one seed
    # replace() returns a BRAND NEW Config rather than editing the shared one, which is why self.base can be safely reused for every run in the sweep.
    def run_one(self, n_cells: int, seed: int, biased: bool = True, save: bool = True) -> SimulationResult:
        cfg = replace(self.base, n_cells=n_cells, seed=seed, biased=biased)
        rng = np.random.default_rng(seed)      # one generator per run, built from the seed and passed down --> the Part 1 decision(allows any single run to bereproducible on its own)
        result = run_simulation(make_field(self.field_name), cfg, rng)
        if save:
            result.save(result_path(self.out_dir, self.field_name, n_cells, seed, biased))
        return result

    # The N = 10, 100, 1000 sweep at a fixed seed. A DICT COMPREHENSION runs all three and keys them by N in one line
    def sweep_population(self, sizes: Sequence[int] = POPULATION_SIZES, seed: int = 0,
                         biased: bool = True, save: bool = True) -> Dict[int, SimulationResult]:
        return {n: self.run_one(n, seed, biased, save) for n in sizes}

    # Repeat ONE condition across several seeds. This is what gives the error bars in the figures
    def sweep_seeds(self, n_cells: int, seeds: Iterable[int], biased: bool = True,
                    save: bool = True) -> List[SimulationResult]:
        return [self.run_one(n_cells, s, biased, save) for s in seeds]   # Run the same population size and bias setting once per seed while passing the save choice to run_one
    
    # Reload every saved result for this field, keyed by N, so figures can be rebuilt long after the simulations finished (or on a others computers)
    def load_all(self, biased: bool = True) -> Dict[int, SimulationResult]:
        tag = "biased" if biased else "control"
        files = sorted(self.out_dir.glob(f"{self.field_name}_{tag}_N*.npz"))   # glob() finds every file matching the pattern --> this is exactly why result_path enforces one naming convention
        return {SimulationResult.load(f).n_cells: SimulationResult.load(f) for f in files}
