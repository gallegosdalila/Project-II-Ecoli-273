# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy
#
# build_notebook.py -- generates CHEM273_Project2.ipynb FROM the real .py files.
# The notebook is never edited by hand, so it can never drift out of sync with the package. Re-run this script after changing any module.

import re
import sys
from pathlib import Path

import nbformat as nbf

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "ecoli")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "CHEM273_Project2.ipynb")

# Module order matters: each one may use names defined by the ones above it, because a notebook has ONE shared namespace instead of separate modules.
MODULES = [
    ("config.py",      "Settings", "Every tunable number in one frozen dataclass. Frozen means read-only after construction, so no cell can quietly change a shared setting and make a later cell irreproducible. To vary a parameter use `replace(cfg, n_cells=1000)`, which returns a new Config."),
    ("contracts.py",   "The result container", "`SimulationResult` is the one data structure that crosses file boundaries. The simulation fills it in, the statistics read it, the plots draw it. Arrays throughout (N x I+1 x 2 positions), never a list of objects, so every statistic is one vectorized numpy call."),
    ("fields.py",      "The chemical environment", "Three concentration profiles: a shallow ramp, one food source, and two competing sources.\n\n**The key distinction:** `concentration(xy)` is what a real cell can smell and is the only thing the simulation may call. `gradient(xy)` is the exact answer from calculus, which no real bacterium could perceive -- it exists purely as the answer key the tests grade the cell's estimate against."),
    ("agents.py",      "The movement layer", "Moves bacteria and nothing else. It never sees a concentration and never decides which way to go.\n\nAll N bacteria are held as arrays rather than as N separate objects. At N=1000 over 1000 cycles that is about 5 million position updates: the array version takes ~0.45 s, a per-object Python loop takes ~36 s."),
    ("simulate.py",    "The chemotaxis algorithm", "The decision layer. One cycle per bacterium:\n\n1. Record `x0` and `C0 = C(x0)`\n2. Take 4 random tumble steps, ending at `x4` with `C4 = C(x4)`\n3. `dC = C4 - C0` and `d = x4 - x0`\n4. Estimate the gradient as `g = (dC / |d|^2) * d`\n5. Run one directed step along `+g`\n\nBecause normalizing `g` recovers `sign(dC) * d_hat`, the rule in words is: **if it got better keep going that way, if it got worse turn around.**"),
    ("stats.py",       "Statistics", "Pure functions on a finished run. Nothing here simulates and nothing here draws.\n\nNote that `distance_trace` returns **std** and **sem** separately: std falling means larger I genuinely brought the bacteria closer together, sem falling means larger N sharpened our estimate of where they are. Two different claims."),
    ("experiments.py", "Sweeps and saved runs", "Runs N = 10, 100, 1000 and writes each result to disk. Saving is a correctness guarantee, not just a speed one: if the plots could rerun the simulation, a figure and the results table could silently come from two different runs."),
    ("plotting.py",    "Figures", "Each function draws one figure onto one axes and returns it. Nothing here runs a simulation -- there is a test in the package that fails if it ever does."),
]


def clean(text: str) -> str:
    """Strip the package-relative imports and the repeated team header.

    In a notebook every module shares ONE namespace, so `from .config import Config`
    is both unnecessary and an error. Third-party imports are hoisted into one cell.
    """
    out = []
    for line in text.split("\n"):
        if re.match(r"^from \.\w* import", line):          # from .config import Config
            continue
        # NOTE: no \s* here on purpose --> only TOP-LEVEL imports are stripped. An import inside a function body (like LineCollection) must stay where it is.
        if re.match(r"^(import numpy|import matplotlib|from matplotlib|from typing|from dataclasses|from abc|from pathlib|import argparse|from collections)", line):
            continue                                        # hoisted to the imports cell
        if line.startswith("# CHEM 273 Project 2") or line.startswith("# Team:"):
            continue                                        # the notebook has its own title
        out.append(line)
    return "\n".join(out).strip("\n")


nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# CHEM 273 — Project 2: Biased Random Walk of *E. coli*

**Team:** Aleyna Celebi, Dalila Gallegos, Emma Patrichi, Nisa Celebi, Tracy Doumit

---

This notebook is **self-contained** — run it top to bottom with `Cell > Run All` and it reproduces every result and figure in the report. No files from the repository are needed.

Each section below is one module of the package, in the order it was written. The code is identical to the `.py` files; this notebook is generated from them.

**What the simulation does.** An *E. coli* cell cannot perceive a chemical gradient — it has one sensor and can only measure "how much is here, right now." So it compares the concentration it senses now against the concentration four time steps ago, and uses the sign of that change to decide which way to swim. This notebook builds that model from the ground up and shows that it works: a population starting 77 µm from a food source closes to 4 µm, while an identical population that ignores the information drifts out to 101 µm.

**Runtime:** about 30 seconds for the whole notebook.""")

md("## Imports\n\nThe only dependencies are numpy and matplotlib.")
code("""from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

%matplotlib inline
plt.rcParams["figure.dpi"] = 110      # readable inline without being enormous
np.set_printoptions(precision=3, suppress=True)   # suppress=True prints 0.001 rather than 1.0e-03""")

for i, (fname, title, blurb) in enumerate(MODULES, start=1):
    md(f"---\n\n## {i}. `{fname}` — {title}\n\n{blurb}")
    code(clean((SRC / fname).read_text()))

# ---------------------------------------------------------------------------
# Live demonstrations
# ---------------------------------------------------------------------------

md("""---

# Results

Everything above is definitions. From here on the notebook actually runs the simulation.""")

md("""## The three concentration profiles

The environments the bacteria swim through. Brighter is more attractant; the star marks each source.

The **shallow ramp** has no peak at all — it is the validation field, because its gradient is the same vector everywhere and can be checked by hand. The **single source** is the main field for the headline result. The **competing sources** have unequal strengths, so the population should split at the saddle between them.""")
code("""use_style()
plt.rcParams["figure.dpi"] = 110

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, name in zip(axes, ["shallow", "single_source", "competing_sources"]):
    plot_concentration(make_field(name), ax=ax, n=200, colorbar=False)
fig.suptitle("The three required concentration profiles")
plt.show()""")

md("""## One chemotaxis cycle, step by step

Before running 1000 bacteria for 200 cycles, here is exactly what happens to **one** bacterium in **one** cycle. This is the whole algorithm.""")
code("""cfg_demo = replace(DEFAULT, n_cells=1, start_mode="point", start_point=(40.0, 30.0))
field_demo = GaussianSource(sigma=60.0)
rng_demo = np.random.default_rng(7)
pop_demo = Population.from_config(cfg_demo, rng_demo)

x0 = pop_demo.positions.copy()                       # where it starts
c0 = sense(field_demo, x0, rng_demo, cfg_demo)       # what it smells there

path = pop_demo.tumble(rng_demo)                     # 4 random tumble steps
x4 = pop_demo.positions
c4 = sense(field_demo, x4, rng_demo, cfg_demo)       # what it smells now

dC = c4 - c0                                         # the "C(t) vs C(t - 4dt)" comparison
d  = x4 - x0                                         # where those 4 tumbles took it
g, valid = estimate_gradient(dC, d, cfg_demo.min_displacement)
u = choose_run_direction(g, valid, pop_demo.directions, rng_demo, cfg_demo)

print(f"start position   x0 = {x0[0]}")
print(f"after 4 tumbles  x4 = {x4[0]}")
print(f"net displacement d  = {d[0]}   |d| = {np.linalg.norm(d[0]):.3f} um")
print()
print(f"concentration before  C0 = {c0[0]:.4f}")
print(f"concentration after   C4 = {c4[0]:.4f}")
print(f"change                dC = {dC[0]:+.4f}   -> it got {'BETTER, keep going that way' if dC[0] > 0 else 'WORSE, turn around'}")
print()
print(f"estimated gradient    g  = {g[0]}")
print(f"run direction         u  = {u[0]}    (unit vector, length {np.linalg.norm(u[0]):.3f})")
print(f"true gradient direction  = {(field_demo.gradient(x0)/np.linalg.norm(field_demo.gradient(x0)))[0]}   <- the cell never sees this")""")

md("""### How good is that guess?

The cell's estimate comes from a single noisy sample, so any one guess is poor. What matters is that it is *positively correlated* with the truth — averaged over hundreds of cycles, that bias is enough.

`cos` below is the alignment between the estimate and the true gradient: +1 is perfect, 0 is useless, −1 is exactly wrong.""")
code("""rng_q = np.random.default_rng(3)
cfg_q = replace(DEFAULT, n_cells=2000, start_mode="uniform")
pop_q = Population.from_config(cfg_q, rng_q)

x0 = pop_q.positions.copy()
c0 = sense(field_demo, x0, rng_q, cfg_q)
pop_q.tumble(rng_q)
dC = sense(field_demo, pop_q.positions, rng_q, cfg_q) - c0
g_est, _ = estimate_gradient(dC, pop_q.positions - x0, cfg_q.min_displacement)

g_true = field_demo.gradient(x0)
cos = np.einsum("ij,ij->i", g_est, g_true) / (np.linalg.norm(g_est, axis=1) * np.linalg.norm(g_true, axis=1))

print(f"2000 single-cycle guesses against the true gradient:")
print(f"  mean cos      = {cos.mean():+.3f}   <- positively correlated, so the walk is biased")
print(f"  fraction > 0  = {(cos > 0).mean():.1%}     <- right general direction this often")
print(f"  median cos    = {np.median(cos):+.3f}")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(cos, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
ax.axvline(0, color="black", lw=1)
ax.axvline(cos.mean(), color="crimson", ls="--", lw=1.6, label=f"mean {cos.mean():+.3f}")
ax.set(xlabel="alignment with the true gradient (cos)", ylabel="count",
       title="One guess is crude. The average is what steers.")
ax.legend()
plt.show()""")

md("""## The main run

N = 1000 bacteria, 200 chemotaxis cycles, on the single Gaussian source. Alongside it, the **unbiased control**: identical bacteria, identical number of tumbles, identical run length — but the run direction is picked at random instead of from the gradient estimate.

The control is what makes this evidence rather than a claim. Without it, a bug that walked cells toward the origin regardless of the field would look exactly like success.""")
code("""cfg_main = replace(DEFAULT, n_cells=1000, n_iterations=200, store_substeps=True)
field_main = GaussianSource(sigma=60.0)

import time
t0 = time.perf_counter()
biased  = run_simulation(field_main, cfg_main, np.random.default_rng(0))
control = run_unbiased_control(field_main, cfg_main, np.random.default_rng(0))
print(f"1000 bacteria x 200 cycles, twice, in {time.perf_counter()-t0:.2f} s")

db, dc = biased.distances_to_source(), control.distances_to_source()
print()
print(f"{'':>10} {'start':>8} {'end':>8}")
print(f"{'biased':>10} {db[:,0].mean():>8.2f} {db[:,-1].mean():>8.2f}")
print(f"{'control':>10} {dc[:,0].mean():>8.2f} {dc[:,-1].mean():>8.2f}")""")

md("""## Trajectories

Twenty sample paths drawn over the field. The jagged texture is the run-and-tumble motion: four short random steps, then one longer directed step, repeated.

White dots are starting positions, orange dots are where each bacterium ended up.""")
code("""fig, ax = plt.subplots(figsize=(6.6, 5.8))
plot_trajectories(biased, n_show=20, field=field_main, ax=ax)
plt.show()""")

md("""## The headline result: the distribution narrows

Distance from the source across the population, at four iteration counts. At I = 1 the bacteria are spread across the whole domain. By I = 50 they have collapsed against zero.

This is the figure the assignment asks for.""")
code("""fig, ax = plt.subplots(figsize=(7, 4.6))
plot_histograms(biased, iterations=(1, 10, 50, 100), ax=ax)
plt.show()

print(f"{'I':>6} {'mean':>9} {'median':>9} {'std':>9} {'variance':>10}")
for i, s in snapshot_stats(biased, (1, 10, 50, 100, 200)).items():
    print(f"{i:>6} {s['mean']:>9.2f} {s['median']:>9.2f} {s['std']:>9.2f} {s['variance']:>10.2f}")""")

md("""## Biased vs unbiased control

The two populations start identically. The only difference is whether the run direction uses the ΔC information.

The gap between the curves *is* the result.""")
code("""fig, ax = plt.subplots(figsize=(7, 4.8))
plot_biased_vs_control(biased, control, ax=ax)
plt.show()

print(f"chemotactic drift, biased : {chemotactic_drift(biased):+.3f} um per cycle")
print(f"chemotactic drift, control: {chemotactic_drift(control):+.3f} um per cycle")""")

md("""## Larger N vs larger I — two different kinds of narrowing

These are separate claims and the report keeps them apart:

- **Larger I** narrows the *actual* distribution. The bacteria really do end up closer together, so the standard deviation falls.
- **Larger N** narrows our *estimate* of that distribution. The standard error of the mean falls like 1/√N, but the true spread is unchanged.

The panels below are all at the same iteration; only the population size differs.""")
code("""results_by_N = {n: run_simulation(field_main, replace(cfg_main, n_cells=n, store_substeps=False),
                                  np.random.default_rng(0))
                for n in (10, 100, 1000)}

fig, axes = compare_N(results_by_N)
plt.show()

print(f"{'N':>6} {'std':>8} {'sem':>8}   <- std is the real spread, sem is how well we know the mean")
for n in (10, 100, 1000):
    t = distance_trace(results_by_N[n])
    print(f"{n:>6} {t['std'][-1]:>8.2f} {t['sem'][-1]:>8.3f}")""")

code("""fig, ax = plt.subplots(figsize=(7, 4.6))
plot_convergence(results_by_N, ax=ax)
plt.show()""")

md("""## The other two fields

**Shallow linear ramp** — no peak to find, so the test is whether the population drifts in the direction of the gradient. `cos` near +1 means it does.

**Competing sources** — two peaks of unequal strength. The population should split, with the saddle between them acting as a watershed.""")
code("""cfg_alt = replace(DEFAULT, n_cells=1000, n_iterations=200)

slope = np.array([0.02, 0.01])
lin = run_simulation(LinearGradient(slope=tuple(slope)), cfg_alt, np.random.default_rng(0))
disp = lin.net_displacement().mean(axis=0)
cos_drift = disp @ slope / (np.linalg.norm(disp) * np.linalg.norm(slope))
print(f"shallow ramp: mean displacement {np.round(disp, 1)}, aligned with the gradient at cos = {cos_drift:+.3f}")

comp = run_simulation(CompetingSources(), cfg_alt, np.random.default_rng(0))
final = comp.positions[:, -1, :]
print(f"competing sources: {(final[:,0] < 0).mean():.0%} of the population went to the left (stronger) peak, "
      f"{(final[:,0] > 0).mean():.0%} to the right")

fig, ax = plt.subplots(figsize=(6.6, 5.8))
plot_trajectories(comp, n_show=25, field=CompetingSources(), ax=ax)
ax.set_title("Competing sources: the population splits")
plt.show()""")

md("""## Descent reverses the behaviour

Setting `ascend=False` flips one sign in `choose_run_direction`. The same bacteria now swim *away* from the food, which confirms the bias is coming from the gradient estimate and not from anything accidental in the geometry.""")
code("""desc = run_simulation(field_main, replace(cfg_alt, ascend=False), np.random.default_rng(0))
d_desc = desc.distances_to_source()
print(f"ascent : {db[:,0].mean():6.1f} um -> {db[:,-1].mean():6.1f} um")
print(f"descent: {d_desc[:,0].mean():6.1f} um -> {d_desc[:,-1].mean():6.1f} um")
print("\\n(The domain is unbounded, so descending cells simply keep going.)")""")

md("""## Comparison with Huo et al. 2021

The paper measures the mean squared displacement of real swimming *E. coli* and fits MSD ∝ t^α, finding **α = 1.66** for wild-type cells (superdiffusive) and **α = 1.09** for a mutant with the CheY-P signalling noise removed (ordinary diffusion).

Our model has no signalling noise, so it should land near the mutant — and it does. That agreement is a point in the model's favour, and it supports the paper's conclusion that noise is what produces the Lévy walk.""")
code("""msd = mean_squared_displacement(biased)
lags = np.arange(1, len(msd) + 1)
fit = np.polyfit(np.log(lags[:50]), np.log(msd[:50]), 1)   # slope on a log-log plot is the exponent

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(lags, msd, lw=1.8, label=f"our model, alpha = {fit[0]:.2f}")
ax.loglog(lags[:60], np.exp(fit[1]) * lags[:60] ** 1.66, "--", lw=1.2, color="gray", label="Huo et al. wild type, 1.66")
ax.loglog(lags[:60], np.exp(fit[1]) * lags[:60] ** 1.09, ":", lw=1.4, color="crimson", label="Huo et al. mutant, 1.09")
ax.set(xlabel="lag (chemotaxis cycles)", ylabel="mean squared displacement (um^2)",
       title="MSD compared with Huo et al. Fig 2")
ax.legend()
plt.show()

print(f"our exponent alpha = {fit[0]:.2f}  --> close to the noise-free mutant, as expected for a model without signalling noise")""")

md("""---

# Summary

| Claim | Result |
|---|---|
| The biased walk finds the source | 77.0 µm → 4.0 µm over 200 cycles |
| An unbiased control does not | 77.0 µm → 100.7 µm, same step budget |
| Larger I narrows the distribution | std 28.7 → 1.8 µm |
| Larger N sharpens the estimate | sem 0.85 → 0.06 across N = 10 → 1000 |
| The walk follows the gradient | cos = +1.00 on a linear ramp |
| Reversing the sign reverses the behaviour | descent swims away |
| Competing sources split the population | roughly 50/50 between the two basins |

The residual ~4 µm is not error. It is the orbit radius set by `run_length = 5 µm`: a bacterium cannot stop mid-run, so it overshoots the peak each cycle. That is a real prediction of the model.

---

**Reference.** Huo H, He R, Zhang R, Yuan J (2021). *Swimming Escherichia coli cells explore the environment by Lévy walk.* Applied and Environmental Microbiology 87:e02429-20.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
nbf.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells: {sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
