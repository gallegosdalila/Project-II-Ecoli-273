# CHEM 273 — Project 2: Biased Random Walk of E. coli

**Team:** Aleyna Celebi, Dalila Gallegos, Emma Patrichi, Nisa Celebi, Tracy Doumit

A vectorized simulation of *E. coli* chemotaxis. Each bacterium alternates four random tumble steps with one directed run, choosing the run direction by comparing the concentration it senses now against the concentration four time steps ago — the same information a real cell has.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires numpy, matplotlib, pytest.

---

## How to run

Two commands reproduce every number and every figure from a clean checkout.

### 1. Run the simulations

```bash
python -m ecoli.main --field single_source --n 10 100 1000 --iters 200 --control --store-substeps
```

**Inputs**

| flag | meaning | default |
|---|---|---|
| `--field` | which concentration profile: `shallow`, `linear`, `single_source`, `competing_sources` | `single_source` |
| `--n` | population sizes to run, space separated | `10 100 1000` |
| `--iters` | chemotaxis cycles per bacterium (I) | `200` |
| `--seed` | random seed | `0` |
| `--seeds` | several seeds instead of one, for error bars | — |
| `--control` | also run the unbiased control | off |
| `--descend` | move *down* the gradient instead of up | off |
| `--store-substeps` | keep every tumble position (needed for the trajectory figure) | off |
| `--boundary` | `none`, `reflect`, or `wrap` | `none` |
| `--out` | folder for the `.npz` files | `results` |

**Expected output**

```
field=single_source  iters=200  direction=ascent
     N  seed     mode       d0       dI    drift
------------------------------------------------
    10     0   biased    85.95     5.07    0.404
    10     0  control    85.95   104.83   -0.094
   100     0   biased    81.11     3.97    0.386
   100     0  control    81.11    97.92   -0.084
  1000     0   biased    77.04     4.04    0.365
  1000     0  control    77.04   100.72   -0.118

distance-from-source statistics, N = 1000
     I      mean    median       std        var
     1     74.10     77.30     28.73     825.38
    10     47.98     49.56     27.01     729.78
    50      4.08      4.01      1.83       3.35
   100      3.97      3.89      1.77       3.12

results written to results/
next:  python -m ecoli.figures --results results --field single_source
```

`d0` is the mean starting distance from the source, `dI` the mean final distance, `drift` the µm closed per cycle. The biased population closes from 77 µm to 4 µm; the unbiased control, with the identical step budget, drifts out to 101 µm. Runtime is about 6 seconds.

### 2. Build the figures

```bash
python -m ecoli.figures --results results --field single_source
```

**Expected output**

```
figures/fig1_fields.png                      the three concentration profiles
figures/fig2_trajectories_single_source.png  20 sample paths over the field
figures/fig3_narrowing_single_source.png     distance histograms at I = 1, 10, 50, 100
figures/fig4_compare_N_single_source.png     the same histogram at N = 10, 100, 1000
figures/fig5_convergence_single_source.png   mean distance vs iteration, one band per N
figures/fig6_control_single_source.png       biased vs unbiased control
figures/fig7_population_split_{field}.png    which source each bacterium
                                             converged on (only for multi-source fields, such as competing_sources)
```

If a needed `.npz` is missing this raises `FileNotFoundError` and prints the exact command to produce it — it will never quietly rerun a simulation, because then a figure and the results table could come from two different runs.

### 3. Run the tests

```bash
pytest
```

Expected: **89 passed**.

---

## Bug fix: nearest-source measurement for `competing_sources`

`distance_trace` and `snapshot_stats` always supported measuring distance to whichever source is closest (`nearest=True`), but `chemotactic_drift`, `fraction_within`, and the CLI itself did not: they always measured distance to source 0, with no way to opt out. For `competing_sources`, that silently misread a bacterium that converged on the *second* peak as having failed to converge at all.

`main.py` now auto-detects `nearest` from the field's source count, so single-source fields are completely unaffected. Before/after on `competing_sources`, N = 300, same seed:

| | `dI` (mean final distance) | variance |
|---|---|---|
| Before (source 0 only) | 53.55 | 2299.75 |
| After (nearest source) | 4.08 | 3.35 |

Same simulation, same seed. The population was converging tightly the whole time; the old measurement just couldn't see it.

---

## Files

| File | What it does |
|---|---|
| `ecoli/config.py` | Every tunable number in one frozen dataclass, so five files cannot drift apart. |
| `ecoli/fields.py` | The chemical environment: C(x, y) for the shallow ramp, one source, and two competing sources. |
| `ecoli/agents.py` | The movement layer. Moves bacteria; never decides which way. |
| `ecoli/simulate.py` | The chemotaxis algorithm: estimates the gradient from ΔC and runs the biased walk. |
| `ecoli/contracts.py` | `SimulationResult`, the one data structure that crosses file boundaries, plus save/load. |
| `ecoli/stats.py` | Statistics on a finished run: distances, means, variance, drift, MSD. |
| `ecoli/experiments.py` | Sweeps N and seeds, writes each run to `results/`. |
| `ecoli/plotting.py` | Draws one figure onto one axes. Never runs a simulation. |
| `ecoli/figures.py` | Regenerates all six report figures from saved results. |
| `ecoli/main.py` | Command line entry point. |
| `ecoli/animation.py` | Optional! Saves a frame-by-frame GIF of a finished run. Not imported by anything else. |
| `tests/` | 89 tests across the six modules. |

Data flow: `main.py` → `experiments.py` → `simulate.py` → `results/*.npz` → `figures.py` → `figures/*.png`

---

## The algorithm

One chemotaxis cycle, per bacterium:

1. Record the position `x0` and concentration `C0 = C(x0)`.
2. Take 4 random tumble steps, ending at `x4` with `C4 = C(x4)`.
3. `ΔC = C4 − C0` and `d = x4 − x0`.
4. Estimate the gradient as `g = (ΔC / |d|²) · d`, the best estimate available from a single directional sample.
5. Run one directed step along `+g`.

Because normalizing `g` recovers `sign(ΔC)·d̂`, the rule in plain language is: **if it got better, keep going that way; if it got worse, turn around.**

A real cell cannot perceive a gradient. It has one sensor and can only compare across time. `fields.gradient()` computes the exact gradient analytically, but it exists **only for tests**, as the answer key that `estimate_gradient()` is graded against. It is never called by the simulation.

---

## Every class and function

### `config.py`

| Name | What it does |
|---|---|
| `Config` | Frozen dataclass holding all 20 simulation parameters. |
| `Config.as_dict` | Returns every field as a plain dict, so a saved run records its own settings. |
| `DEFAULT`, `SMALL`, `MEDIUM`, `LARGE` | Preset configs so nobody retypes the same numbers. |
| `POPULATION_SIZES`, `SNAPSHOT_ITERATIONS` | The N and I values the assignment asks for. |

### `fields.py`

| Name | What it does |
|---|---|
| `ConcentrationField` | Abstract base class; every field must supply `concentration` and `gradient`. |
| `._as_points` | Validates input shape so bad input fails at the door, not deep in the math. |
| `.numerical_gradient` | Estimates the gradient by finite difference; used only to check the analytical one. |
| `.meshgrid` | Builds (X, Y, C) grids for contour plots. |
| `LinearGradient` | Shallow ramp `C = c0 + g·(x − x_ref)`; its gradient is constant, so it can be checked on paper. |
| `GaussianSource` | One food source, `C = A exp(−|x − x_s|²/2σ²)`; gradient points inward. |
| `CompetingSources` | Sum of several Gaussians; the population splits at the saddle between peaks. |
| `make_field` | Builds a field by name, raising with the valid options on a typo. |

### `agents.py`

| Name | What it does |
|---|---|
| `random_unit_vectors` | N uniformly random 2D unit vectors in one call. |
| `random_unit_grid` | Same, with any leading shape: draws all N × 4 tumble directions at once. |
| `normalize` | Scales each row to length 1, leaving zero rows as zero rather than NaN. |
| `apply_boundary` | Applies the domain rule: unbounded, reflect, or wrap. |
| `initial_positions` | Places the starting population: uniform, point, or ring. |
| `Population` | All N bacteria held as arrays, roughly 80× faster than one object per cell. |
| `Population.from_config` | Builds a population straight from a `Config`. |
| `Population.tumble` | Takes 4 random steps for every bacterium using one cumulative sum. |
| `Population.run` | Moves every bacterium one directed step; a zero vector means stay put. |

### `simulate.py`

| Name | What it does |
|---|---|
| `estimate_gradient` | Estimates ∇C from one ΔC per bacterium, the core of the whole project. |
| `directional_derivative` | ΔC per unit distance travelled; used by the tests. |
| `choose_run_direction` | Turns the estimate into a unit run direction, with a fallback if the tumbles cancelled. |
| `sense` | Reads the field, optionally with sensor noise. |
| `chemotaxis_cycle` | Four tumbles, one estimate, one run, for all N bacteria at once. |
| `run_simulation` | Runs N bacteria for I cycles and returns a filled `SimulationResult`. |
| `run_unbiased_control` | The same run with a random direction, the baseline that makes the result evidence. |

### `contracts.py`

| Name | What it does |
|---|---|
| `SimulationResult` | Holds everything one run produced: positions, concentrations, ΔC, and the config used. |
| `.n_cells`, `.n_iterations` | Shape helpers so callers never count axes by hand. |
| `.distances_to_source` | Distance from every bacterium to the source at every iteration, the headline measurement. |
| `._distances_to_each_source` | Distance from every bacterium to every source, shape (N, I+1, S); shared by the two rows below so they can't drift apart. |
| `.distances_to_nearest_source` | Same, but to whichever source is closest; needed for competing sources. |
| `.nearest_source_index` | Which source is closest to each bacterium. Replaces an inline check that only worked when sources sat left/right of x=0; works for any layout. |
| `.net_displacement` | Start-to-finish vector per bacterium, which shows the net drift. |
| `.validate` | Checks every array shape agrees before the result is handed to another file. |
| `.save` / `.load` | Writes and reads one compressed `.npz`, so figures never rerun a simulation. |

### `stats.py`

| Name | What it does |
|---|---|
| `distance_stats` | Mean, median, std, variance and quartiles for one snapshot. |
| `distance_trace` | The same statistics at every iteration, returning std and SEM separately. |
| `snapshot_stats` | `distance_stats` at each requested I, skipping any past the end of the run. |
| `mean_squared_displacement` | MSD vs lag, for comparison with Huo et al. Fig 2. |
| `chemotactic_drift` | Mean µm closed per cycle, the cleanest single number separating biased from control. `nearest=True` measures against whichever source is closest instead of always source 0. |
| `fraction_within` | Share of the population inside a given radius of the source. Same nearest fix as above. |

### `experiments.py`

| Name | What it does |
|---|---|
| `result_path` | One naming rule for saved runs, so figures can find any file without guessing. |
| `ExperimentRunner` | Runs the sweeps and writes each result to disk. |
| `.run_one` | One simulation at one N and one seed. |
| `.sweep_population` | The N = 10, 100, 1000 sweep. |
| `.sweep_seeds` | Repeats one condition across seeds, which gives the error bars. |
| `.load_all` | Reloads every saved run for a field, keyed by N. |

### `plotting.py`

| Name | What it does |
|---|---|
| `STYLE`, `use_style` | One shared house style so every figure in the report matches. |
| `_ax` | Uses the caller's axes or makes a new one, so each function works standalone or as a panel. |
| `plot_concentration` | Filled contour map of C(x, y) with the sources marked. |
| `plot_trajectories` | Sample paths over the field, start and end marked. |
| `plot_histograms` | Distance-from-source histograms at several I, the headline figure. |
| `plot_convergence` | Mean distance vs iteration, one shaded band per N. |
| `compare_N` | One histogram panel per N at a fixed I. |
| `plot_biased_vs_control` | The validation figure: biased against the unbiased control. |
| `plot_population_split` | Bar chart of which source each bacterium ended up closest to. Works for any number of sources; built on `nearest_source_index`. |

### `figures.py`

| Name | What it does |
|---|---|
| `_load` | Loads one saved run, or fails with the exact command to produce it. |
| `figure_fields` | Figure 1: the three concentration profiles. |
| `figure_trajectories` | Figure 2: sample paths over the field. |
| `figure_narrowing` | Figure 3: the distance distribution tightening as I grows. |
| `figure_population_sweep` | Figures 4 and 5: the N comparison and the convergence curves. |
| `figure_control` | Figure 6: biased vs unbiased control. |
| `figure_population_split` | Figure 7: population split across sources. Only built for fields with more than one source. |
| `build_all` | Builds all six figures in order, plus figure 7 automatically when the field has multiple sources. |

### `main.py`

| Name | What it does |
|---|---|
| `build_parser` | Defines every command line flag. |
| `main` | Runs the sweep, prints the summary and statistics tables, points at the next command. |

### `animation.py` (optional)

| Name | What it does |
|---|---|
| `animate_walk` | Saves a GIF: one frame per completed chemotaxis cycle, source(s) marked, current cycle shown as text. Axis limits are computed from the real positions, not just the configured domain size, so an unbounded/control run can't get clipped off-screen. `tqdm` is optional, falls back to a plain loop if it isn't installed. |
| `main` | Standalone CLI: runs one simulation and saves an animation from it. |

---

## Live animation

```bash
python -m ecoli.animation --field competing_sources --n 30 --n-show 20 --iters 100 --save walk.gif
```

Saves a GIF: bacteria as moving dots, source(s) marked distinctly, current cycle shown as on-plot text. One frame per completed chemotaxis cycle, written with a plain `for` loop (not `matplotlib.animation.FuncAnimation`) so it's easy to read top to bottom. Lives in `ecoli/animation.py`, kept separate from `plotting.py` on purpose: `plotting.py`'s own test enforces "never runs a simulation, never needs more than a plain Axes," and an animation needs a video writer and (optionally) `tqdm`, which is different enough machinery to warrant its own file. Nothing else in the project imports it, so it's entirely optional: the rest of the pipeline and test suite are unaffected whether or not `tqdm`/`pillow` are installed.

No extra install needed: Pillow ships as one of matplotlib's own dependencies, and `tqdm` is optional; the progress bar just doesn't appear if it's missing.

---

## Reference

Huo H, He R, Zhang R, Yuan J (2021). *Swimming Escherichia coli cells explore the environment by Lévy walk.* Applied and Environmental Microbiology 87:e02429-20.

Used for the run speed (~10 µm/s) and as the comparison for our MSD exponent. Our noise-free model gives α ≈ 0.88, close to their signalling-noise-free mutant (1.09) rather than wild type (1.66), consistent with their conclusion that signalling noise is what produces superdiffusion.

## Implementation references

- Matplotlib Animation API: `PillowWriter`, `MovieWriter.saving()`, `grab_frame()`
  https://matplotlib.org/stable/api/animation_api.html
  Used to write each rendered frame directly to a GIF, without holding every frame in memory at once.

- Matplotlib `PathCollection.set_offsets()`
  https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.PathCollection.set_offsets
  Used to move the bacteria scatter points to their new positions each frame, instead of redrawing the whole plot.

- tqdm
  https://tqdm.github.io/
  Wraps the frame-writing loop to show rendering progress; optional, animation still works without it.
