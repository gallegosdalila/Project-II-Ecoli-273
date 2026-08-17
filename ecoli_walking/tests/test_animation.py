import numpy as np
import pytest

from typing import Optional
from tqdm import tqdm
from ecoli import animation as A
from ecoli.config import Config
from ecoli.fields import GaussianSource, CompetingSources
from ecoli.simulate import run_simulation, run_unbiased_control


CFG = Config(n_cells=20, n_iterations=15, start_mode="uniform")


def test_animate_walk_writes_a_real_file(tmp_path):
    r = run_simulation(GaussianSource(), CFG, np.random.default_rng(0))
    out = tmp_path / "walk.gif"
    A.animate_walk(r, str(out), field=GaussianSource(), n_show=10)
    assert out.exists()
    assert out.stat().st_size > 1_000, "gif looks empty"


def test_animate_walk_has_one_frame_per_cycle(tmp_path):
    from PIL import Image
    r = run_simulation(GaussianSource(), CFG, np.random.default_rng(0))
    out = tmp_path / "walk.gif"
    A.animate_walk(r, str(out), field=GaussianSource(), n_show=10)
    img = Image.open(out)
    assert img.n_frames == r.n_iterations + 1   # +1 for the starting position


def test_animate_walk_works_without_a_field(tmp_path):
    #field= is optional, without one only the bare source marker(s) should still show
    r = run_simulation(GaussianSource(), CFG, np.random.default_rng(0))
    out = tmp_path / "walk.gif"
    A.animate_walk(r, str(out), field=None, n_show=5)
    assert out.exists()


def test_animate_walk_respects_n_show_cap():
    r = run_simulation(GaussianSource(), CFG, np.random.default_rng(0))
    n_show = min(10_000, r.n_cells) # mirrors the clamp inside animate_walk
    assert n_show == r.n_cells# never more than the population actually has


def test_animate_walk_works_for_multi_source_fields(tmp_path):
    field = CompetingSources(centers=((-50.0, 0.0), (50.0, 0.0)))
    r = run_simulation(field, CFG, np.random.default_rng(0))
    out = tmp_path / "walk.gif"
    A.animate_walk(r, str(out), field=field, n_show=10)
    assert out.exists()


#validate inputs actually raise, not just get silently clamped
def test_animate_walk_rejects_bad_n_show(tmp_path):
    r = run_simulation(GaussianSource(), CFG, np.random.default_rng(0))
    with pytest.raises(ValueError):
        A.animate_walk(r, str(tmp_path / "walk.gif"), n_show=0)


def test_animate_walk_rejects_bad_fps(tmp_path):
    r = run_simulation(GaussianSource(), CFG, np.random.default_rng(0))
    with pytest.raises(ValueError):
        A.animate_walk(r, str(tmp_path / "walk.gif"), fps=0)


# a fixed axis limit from config can clip bacteria that actually wander past domain_half_width (e.g. an unbounded control run).
#axis limits are computed from the real positions instead, so this should never clip
def test_animate_walk_axis_limits_cover_bacteria_that_leave_the_configured_domain(tmp_path):
    cfg = Config(n_cells=15, n_iterations=60, start_mode="uniform", boundary="none", domain_half_width=50.0)
    r = run_unbiased_control(GaussianSource(sigma=20.0), cfg, np.random.default_rng(0))
    max_pos = np.abs(r.positions).max()
    assert max_pos > cfg.domain_half_width, "test setup didn't actually leave the domain, pick a different seed"

    out = tmp_path / "walk.gif"
    A.animate_walk(r, str(out), field=None, n_show=15)
    assert out.exists()
