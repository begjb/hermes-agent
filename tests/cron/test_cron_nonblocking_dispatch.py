"""Tests for the non-blocking cron tick dispatch path.

Regression coverage for the serial-blocking ticker starvation bug
(jetminds incident 2026-06-02): a slow ``no_agent=False`` LLM cron that runs
for minutes must NOT hold the ticker thread, or fast ``no_agent`` scan crons
on a short grace window get fast-forward-skipped forever.

The fix decouples job *execution* from the lock-protected due-detection +
``advance_next_run`` critical section: in dispatch-only mode ``tick()`` returns
promptly after handing due jobs to a persistent background executor.

Covers:

* ``tick(blocking=True)`` (default for manual/CLI) still waits for jobs and
  returns the executed count — preserves exit codes and determinism.
* ``tick(blocking=False)`` returns promptly even when a job sleeps, and the
  job still runs to completion on the background executor.
* ``advance_next_run`` is called under the lock BEFORE dispatch in both modes,
  so a slow job cannot be re-picked by a subsequent tick (at-most-once).
* A slow job dispatched non-blocking does not delay a second tick that picks up
  a different fast job.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME so jobs/state don't leak across tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    yield home

    # Tear down any persistent executor the test started so threads don't leak.
    try:
        cron.scheduler.shutdown_persistent_cron_executor(wait=True)
    except Exception:
        pass


def _make_interval_job(minutes=5, name="slow"):
    """Create a recurring interval job and backdate next_run_at so it is due now.

    A fresh interval job's next_run_at is now+period (future). We pull it ~10s
    into the past — late, but within the 150s grace for a 5m interval — so the
    very first tick treats it as due without tripping the fast-forward skip.
    """
    from cron.jobs import create_job, load_jobs, save_jobs
    from datetime import timedelta
    from cron.scheduler import _hermes_now

    job = create_job(
        prompt="noop",
        schedule=f"every {minutes}m",
        name=name,
    )
    past = (_hermes_now() - timedelta(seconds=10)).isoformat()
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job["id"]:
            j["next_run_at"] = past
    save_jobs(jobs)
    return job


def test_blocking_tick_waits_and_counts(hermes_env):
    """Default blocking mode runs jobs inline and returns the executed count."""
    import cron.scheduler as sched

    job = _make_interval_job()
    ran = []

    def fake_run_job(j):
        ran.append(j["id"])
        return (True, "out", "final", None)

    with patch.object(sched, "run_job", side_effect=fake_run_job), \
         patch.object(sched, "_deliver_result", return_value=None):
        n = sched.tick(verbose=False, blocking=True)

    assert n == 1
    assert ran == [job["id"]]


def test_nonblocking_tick_returns_before_slow_job_finishes(hermes_env):
    """Dispatch-only mode must return promptly even if the job runs for seconds."""
    import cron.scheduler as sched

    _make_interval_job(name="slow")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_run_job(j):
        started.set()
        # Hold the "execution" until the test lets it go.
        release.wait(timeout=10)
        finished.set()
        return (True, "out", "final", None)

    with patch.object(sched, "run_job", side_effect=slow_run_job), \
         patch.object(sched, "_deliver_result", return_value=None):
        t0 = time.monotonic()
        dispatched = sched.tick(verbose=False, blocking=False)
        elapsed = time.monotonic() - t0

        # The tick returned how many jobs it dispatched, without waiting.
        assert dispatched == 1
        # Returned promptly — well under the job's hold time.
        assert elapsed < 2.0, f"tick blocked {elapsed:.2f}s on a slow job"
        # The job actually started on the background executor.
        assert started.wait(timeout=5), "dispatched job never started"
        assert not finished.is_set(), "job finished too early; tick must not have waited"

        # Let it complete and confirm it really ran.
        release.set()
        assert finished.wait(timeout=5), "dispatched job never finished"


def test_advance_happens_before_dispatch_prevents_repick(hermes_env):
    """A slow job dispatched non-blocking must not be re-picked by the next tick."""
    import cron.scheduler as sched

    _make_interval_job(name="slow")
    release = threading.Event()
    call_count = {"n": 0}

    def slow_run_job(j):
        call_count["n"] += 1
        release.wait(timeout=10)
        return (True, "out", "final", None)

    with patch.object(sched, "run_job", side_effect=slow_run_job), \
         patch.object(sched, "_deliver_result", return_value=None):
        first = sched.tick(verbose=False, blocking=False)
        assert first == 1
        # Immediately tick again while the first job is still "running".
        second = sched.tick(verbose=False, blocking=False)
        # next_run_at was advanced under the lock before dispatch, so the job
        # is no longer due — the second tick finds nothing.
        assert second == 0, "job was re-picked before its next_run advanced"

        release.set()
        sched.shutdown_persistent_cron_executor(wait=True)
        # Job ran exactly once despite two ticks.
        assert call_count["n"] == 1
