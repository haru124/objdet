"""
profiler/profiler_utils.py

PyTorch profiler integration.

Usage in training loop:
    with build_profiler(cfg.profiler) as prof:
        for batch_idx, batch in enumerate(loader):
            train_step(batch)
            prof.step()   # tells profiler to advance its schedule

The profiler automatically starts/stops tracing based on the schedule and
writes Chrome-trace JSON files to output_dir.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.profiler as profiler

from objdet.entity.config_entity import ProfilerConfig


def build_profiler(prof_cfg: ProfilerConfig):
    """
    Return a torch.profiler.profile context manager configured from *prof_cfg*.

    If profiler.enabled is False, returns a no-op context manager instead so
    training code doesn't need to branch.

    Args:
        prof_cfg: ProfilerConfig with wait/warmup/active/output_dir settings.
    """
    if not prof_cfg.enabled:
        return _noop_profiler()

    output_dir = Path(prof_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule = profiler.schedule(
        wait=prof_cfg.wait,
        warmup=prof_cfg.warmup,
        active=prof_cfg.active,
        repeat=1,
    )

    trace_handler = profiler.tensorboard_trace_handler(str(output_dir))

    return profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,        # Record tensor shapes (adds small overhead)
        profile_memory=True,       # Track memory allocations
        with_stack=False,          # Python call stack (expensive; enable for debugging)
    )


@contextmanager
def _noop_profiler():
    """A context manager that does nothing — stands in for the real profiler."""

    class _NoopProf:
        def step(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    yield _NoopProf()
