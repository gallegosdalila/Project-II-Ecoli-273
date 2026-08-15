# CHEM 273 Project 2 - Biased Random Walk of E. coli
# Team: Aleyna, Dalila, Emma, Nisa, Tracy

# main.py -- the command line front end. This is how anybody, including a grader with a fresh clone, actually runs the project.

# THIS FILE IS DELIBERATELY THIN: parse the arguments, build a Config, run the sweep, print a summary table, stop.
    # All the real work lives in the other files. If you are ever tempted to put an algorithm in here, it belongs somewhere else.
    # A CLI is a SUMMARY OF THE COMMANDS WE ALREADY TYPED TWENTY TIMES while building the project. That is why it is written LAST -- write it earlier and you are guessing at your own workflow, and you end up with flags nobody uses and no flag for the thing you actually need.

# TYPICAL USE -- these two commands reproduce every number and every figure in the report from a clean checkout:
#     python -m ecoli.main --field single_source --n 10 100 1000 --iters 200 --control --store-substeps
#     python -m ecoli.figures --field single_source

# THIS FILE IS PHASE 7 -- step 37, the last one.

import argparse
from dataclasses import replace

from .config import Config, POPULATION_SIZES, SNAPSHOT_ITERATIONS
from .experiments import ExperimentRunner
from .fields import FIELDS
from .stats import chemotactic_drift, snapshot_stats


# argparse reads the words typed after the filename and turns them into an object with attributes.
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ecoli", description="Simulate the biased random walk of E. coli.")
    ap.add_argument("--field", default="single_source", choices=sorted(FIELDS))   # choices= makes argparse REJECT a typo and print the valid options for us, instead of failing later with a confusing KeyError
    ap.add_argument("--n", type=int, nargs="+", default=list(POPULATION_SIZES), help="population sizes to run")   # nargs="+" accepts several values at once, so --n 10 100 1000 works
    ap.add_argument("--iters", type=int, default=200, help="chemotaxis cycles per bacterium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="run several seeds instead of one, for error bars")
    ap.add_argument("--out", default="results", help="folder to write the .npz files into")
    ap.add_argument("--descend", action="store_true", help="move DOWN the gradient instead of up")   # action="store_true" makes it a flag with no value: present means True, absent means False
    ap.add_argument("--control", action="store_true", help="also run the unbiased control alongside each condition")
    ap.add_argument("--store-substeps", action="store_true", help="keep every tumble position, needed for the trajectory figure (large files)")
    ap.add_argument("--boundary", default="none", choices=["none", "reflect", "wrap"])
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Build ONE Config from the command line, then let ExperimentRunner vary only n_cells, seed and biased on top of it.
    cfg = replace(Config(), n_iterations=args.iters, ascend=not args.descend,
                  boundary=args.boundary, store_substeps=args.store_substeps)

    runner = ExperimentRunner(args.field, cfg, args.out)
    seeds = args.seeds if args.seeds else [args.seed]

    print(f"field={args.field}  iters={args.iters}  direction={'ascent' if cfg.ascend else 'descent'}")
    header = f"{'N':>6} {'seed':>5} {'mode':>8} {'d0':>8} {'dI':>8} {'drift':>8}"   # d0 is the mean STARTING distance, dI the mean FINAL distance, drift is um closed per cycle
    print(header)
    print("-" * len(header))

    for seed in seeds:
        for n_cells in args.n:
            modes = [True, False] if args.control else [True]   # run the control alongside each condition only when asked, since it doubles the runtime
            for biased in modes:
                r = runner.run_one(n_cells, seed, biased=biased)
                d = r.distances_to_source()
                print(f"{n_cells:>6} {seed:>5} {'biased' if biased else 'control':>8} {d[:, 0].mean():>8.2f} {d[:, -1].mean():>8.2f} {chemotactic_drift(r):>8.3f}")

    # The snapshot table for the largest population --> THIS IS THE RESULTS TABLE that goes in the report.
    # save=False because this run exists only to be printed; the one that gets saved already happened in the loop above.
    largest = runner.run_one(max(args.n), seeds[0], biased=True, save=False)
    print(f"\ndistance-from-source statistics, N = {largest.n_cells}")
    print(f"{'I':>6} {'mean':>9} {'median':>9} {'std':>9} {'var':>10}")
    for i, s in snapshot_stats(largest, SNAPSHOT_ITERATIONS).items():
        print(f"{i:>6} {s['mean']:>9.2f} {s['median']:>9.2f} {s['std']:>9.2f} {s['variance']:>10.2f}")

    print(f"\nresults written to {runner.out_dir}/")
    print(f"next:  python -m ecoli.figures --results {runner.out_dir} --field {args.field}")   # tell the user the NEXT command, so nobody has to remember the two-step workflow
    return 0


if __name__ == "__main__":
    raise SystemExit(main())   # SystemExit passes main()'s return value out as the process exit code, which is what a shell or a CI system checks to decide whether the command succeeded
