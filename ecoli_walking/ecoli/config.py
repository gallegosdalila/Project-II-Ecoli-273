# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# config.py -- every tunable number in the simulation, in one place.
# Part 1 of the assignment says shared parameters live in ONE file everyone imports, instead of five hardcoded copies drifting apart. This will be that file.
# Units are micrometers and seconds throughout, loosely matched to Huo et al. 2021 (their measured run speed was about 10 um/s).

# IMPORTANT NOTE:
# To CHANGE a value, do NOT edit it here!!! 
# Use dataclasses.replace(cfg, n_cells=1000), which returns a brand new Config and leaves the original untouched.

from dataclasses import dataclass, asdict   # dataclass used for writing __init__ for us from the annotations below, so we get a typed importable settings object instead of loose global variables


# frozen = True makes an instance READ-ONLY after it is built, so no module can change a shared setting halfway through a run (making results irreproducible)
class Config:
    """Every tunable number in the simulation."""

    # ---- geometry ----
    domain_half_width: float = 100.0     # domain is the square [-H, H] x [-H, H] in micrometers --> used by apply_boundary and initial_positions

    # ---- motion ----
    # One chemotaxis cycle = n_tumbles random steps, then one directed run.
    tumble_step: float = 1.0             # how far one random tumble step moves a bacterium, um
    run_length: float = 5.0              # how far one directed run step moves a bacterium, um --> LONGER than a tumble on purpose, so the bias wins on average and the cell actually climbs
    n_tumbles: int = 4                   # the sampling window --> the "4 * delta t" the assignment asks us to compare across

    # ---- chemotaxis decision ----
    ascend: bool = True                  # True climbs TOWARD higher concentration. The slide says "descent", which would swim cells away from food
    min_displacement: float = 1e-9       # displacement floor, um --> below this the 4 tumbles cancelled out and dividing by |d|^2 would blow up
    fallback: str = "previous"           # what to do when the 4 tumbles cancel --> "previous" reuses the last run direction, "random" picks a fresh one, "skip" sits still for this cycle

    sensing_noise: float = 0.0           # standard deviation of the noise added to every concentration reading. 0.0 means a PERFECT sensor, which is what we use for the headline result.
    biased: bool = True                  # False runs the UNBIASED CONTROL: same number of tumbles and runs, but the run direction is picked at random instead of from the gradient estimate.

    # ---- population ----
    n_cells: int = 100                   # N --> how many E. coli to simulate
    n_iterations: int = 200              # I, how many chemotaxis cycles each bacterium performs. One cycle = 4 tumbles + 1 run.
    start_mode: str = "uniform"          # where the bacteria begin --> "uniform" scatters them across the whole domain, "point" stacks them all at start_point, "ring" places them on a circle
    start_point: tuple[float, float] = (-70.0, -70.0)   # only read when start_mode is "point"
    start_radius: float = 80.0                          # only read when start_mode is "ring"

    # ---- boundary ----                    #DALILA: where is the "reflect" portion? or am i missing it?
    boundary: str = "none"               # "none" leaves the domain unbounded, "reflect" bounces bacteria off the walls, "wrap" teleports them to the opposite edge
                                         # GROUP DECISION: default is "none". Reflect piles cells against the walls and wrap teleports them, and BOTH distort the distance-from-source histogram, which is our headline figure.

    stop_at_source: bool = False         # False lets bacteria keep moving once they arrive
    capture_radius: float = 5.0          # only read when stop_at_source is True: how close counts as "arrived", um

    # ---- bookkeeping ----
    seed: int = 0                        # random seed --> ONE generator is built from this in main.py and passed down to everything, so a whole run reproduces from one number

    store_substeps: bool = False         # True keeps EVERY tumble position, not just the end of each cycle

    # Plain dict of every field, so a saved run can record the exact settings it used alongside its data
    def as_dict(self) -> dict:
        return asdict(self)
                                            #DALILA: long-hand notation for decorator instead of @dataclass(frozen=True)
Config = dataclass(frozen=True)(Config)   # dataclass(frozen=True) returns a decorator, which is then called on the class --> that is why there are two sets of parentheses


# Presets, so nobody retypes these in five places.
DEFAULT = Config()                                 # the settings every module falls back to
#DALILA: small to large scale simulations with n_cells being over-written form default values for testing
SMALL = Config(n_cells=10)
MEDIUM = Config(n_cells=100)
LARGE = Config(n_cells=1000)

POPULATION_SIZES = (10, 100, 1000)                 # the three N values the assignment asks for
SNAPSHOT_ITERATIONS = (1, 10, 50, 100, 1000)       # the iteration counts we report statistics and draw histograms at.
                                                   # Lives here rather than in figures.py so the RESULTS TABLE and the HISTOGRAM FIGURE always show the same I values. Counts past the end of a run are skipped automatically.
