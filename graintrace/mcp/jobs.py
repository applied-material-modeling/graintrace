# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""In-process background job registry for long-running graintrace runs.

MCP tool calls are request/response, but CPFE / meshing / reconstruction runs
take minutes to hours. Heavy tools therefore submit the work here and return a
``job_id`` immediately; the client polls ``job_status`` (see tools/system.py)
for progress. stdout/stderr from the run is tee'd to a per-job log file under
the workdir so the client can tail it.

This is deliberately simple (daemon threads, in-memory dict). It lives for the
lifetime of the server process -- fine for a single-user chat session driving
one machine, which is the target. It is NOT a distributed queue.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from graintrace.mcp.app import workdir


@dataclass
class Job:
    id: str
    tool: str
    status: str = "pending"          # pending | running | done | error
    submitted: float = field(default_factory=lambda: time.time())
    started: Optional[float] = None
    finished: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    log_path: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = None
            if self.started is not None:
                end = self.finished if self.finished is not None else time.time()
                elapsed = round(end - self.started, 1)
            return {
                "job_id": self.id,
                "tool": self.tool,
                "status": self.status,
                "elapsed_s": elapsed,
                "result": self.result,
                "error": self.error,
                "log_path": self.log_path,
            }


class _Tee(io.TextIOBase):
    """Write to an in-memory buffer and a file at once (best-effort)."""

    def __init__(self, fh):
        self._fh = fh

    def write(self, s):  # noqa: D401
        with contextlib.suppress(Exception):
            self._fh.write(s)
            self._fh.flush()
        return len(s)


_JOBS: Dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def submit(tool: str, fn: Callable[[], Any]) -> Job:
    """Run ``fn`` in a daemon thread, capturing its stdout/stderr to a log file."""
    job_id = uuid.uuid4().hex[:12]
    log_path = workdir() / "jobs" / f"{tool}_{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, tool=tool, log_path=str(log_path))
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    def _run():
        with job._lock:
            job.status = "running"
            job.started = time.time()
        try:
            with open(log_path, "w") as fh:
                tee = _Tee(fh)
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    result = fn()
            with job._lock:
                job.result = result
                job.status = "done"
        except Exception as exc:  # captured, surfaced via job_status
            tb = traceback.format_exc()
            with contextlib.suppress(Exception):
                with open(log_path, "a") as fh:
                    fh.write("\n" + tb)
            with job._lock:
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "error"
        finally:
            with job._lock:
                job.finished = time.time()

    threading.Thread(target=_run, name=f"graintrace-job-{job_id}", daemon=True).start()
    return job


def get(job_id: str) -> Optional[Job]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def all_jobs() -> List[dict]:
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    return [j.snapshot() for j in sorted(jobs, key=lambda j: j.submitted, reverse=True)]


def tail(job_id: str, n: int = 40) -> str:
    job = get(job_id)
    if job is None or not job.log_path or not Path(job.log_path).exists():
        return ""
    with open(job.log_path, "r", errors="replace") as fh:
        lines = fh.readlines()
    return "".join(lines[-n:])
