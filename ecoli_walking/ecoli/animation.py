
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np 
from tqdm import tqdm

from typing import Optional
from .contracts import SimulationResult
from .fields import ConcentrationField
from .plotting import SOURCE_MARKER, plot_concentration


#Save a GIF of n_show bacteria moving toward the source, one frame per completed chemotaxis cycle.
#i field is provided, its heatmap is drawn behind the bacteria.
#otherwise, only the source marker(s) are shown on a plain background.
def animate_walk( result: SimulationResult, save_path: str, field: Optional[ConcentrationField] = None, n_show: int = 20, fps: int = 15,) -> None:
    if n_show < 1:
        raise ValueError("n_show must be at least 1")

    if fps < 1:
        raise ValueError("fps must be at least 1")

    #Never ask for more bacteria than exist.
    n_show = min(n_show, result.n_cells)

    #shape: (n_show, n_iterations + 1, 2)
    #one stored position per completed chemotaxis cycle.
    positions = result.positions[:n_show]
    n_frames = positions.shape[1]

    fig, ax = plt.subplots(figsize=(6, 6))

    #match the plotting domain to the domain used by the simulation.
    half_width = float(result.config.get("domain_half_width", 100.0)) 

    if field is not None:
        #draw concentration heatmap and source marker(s).
        plot_concentration( field, half_width=half_width, ax=ax, colorbar=False,)
    else:
        src = np.atleast_2d(result.sources)

        # show source(s) even without the concentration background.
        ax.scatter(src[:, 0], src[:, 1], **SOURCE_MARKER,)

        ax.set( xlim=(-half_width, half_width), ylim=(-half_width, half_width), xlabel="x (um)", ylabel="y (um)",aspect="equal",)
  
    #  change from frame to frame.
    dots = ax.scatter(positions[:, 0, 0], positions[:, 0, 1], c="white", edgecolors="black", zorder=5,)

    label = ax.text( 0.02, 0.98, "",
        transform=ax.transAxes,
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none",
        ),
    )

    ax.set_title(f"E. coli chemotaxis (N shown = {n_show})")

    # PillowWriter collects frames into an animated GIF.
    writer = PillowWriter(fps=fps)

    try:
        with writer.saving(fig, save_path, dpi=100,):
            # tqdm wraps the actual frame-writing loop, so each progress step
            # corresponds to one frame being drawn and saved.
            for i in tqdm(range(n_frames), desc="rendering frames", unit="frame",) :
                #move bacteria to this cycle's positions.
                dots.set_offsets( positions[:, i])

                #show the current simulation cycle.
                label.set_text(f"cycle {i} / {n_frames - 1}")

                #save the current figure as one GIF frame.
                writer.grab_frame()

    finally:
        #always close the figure, even if GIF writing fails.
        plt.close(fig)

    print(f"saved animation to {save_path}")


#small standalone CLI:
#
# python -m ecoli.animation \
#     --field competing_sources \
#     --n 30 \
#     --n-show 20 \
#     --iters 100 \
#     --save walk.gif
def main(argv=None):

    import argparse
    from dataclasses import replace

    from .config import Config
    from .fields import make_field
    from .simulate import run_simulation

    ap = argparse.ArgumentParser(description=("save frame-by-frame animation of one run.")) 

    ap.add_argument("--field", default="single_source",)
    ap.add_argument("--n", type=int, default=30, help="bacteria to simulate",)
    ap.add_argument( "--n-show", type=int, default=20, help="bacteria to actually animate",)
    ap.add_argument("--iters",type=int, default=100,)
    ap.add_argument("--seed", type=int, default=0,)
    ap.add_argument("--save",  default="animation.gif", help="output GIF path",)
    ap.add_argument("--fps", type=int,default=15, )

    args = ap.parse_args(argv)
    field = make_field(args.field)
    cfg = replace( Config(), n_cells=args.n,  n_iterations=args.iters,)
    result = run_simulation(field, cfg, np.random.default_rng(args.seed),)

    animate_walk(result, args.save, field=field, n_show=args.n_show, fps=args.fps,)


if __name__ == "__main__":
    main()