# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""The confirm-before-run contract shared by every executing tool.

A tool resolves its parameters, then calls :func:`gate`. With ``confirm=False``
(the default) this returns a structured *preview* and runs nothing. With
``confirm=True`` it first checks external dependencies (returning a friendly
"not built yet" message if any are missing) and only then invokes the run
callable. This is the single choke point that enforces "no graintrace
computation runs until the user approves".
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from graintrace.mcp import deps
from graintrace.mcp import jobs


def _dep_report(needs: Sequence[str]) -> List[dict]:
    out = []
    for name in needs:
        s = deps.check(name)
        out.append({"name": s.name, "ok": s.ok, "detail": s.detail})
    return out


def gate(
    *,
    tool: str,
    confirm: bool,
    resolved_params: Dict[str, Any],
    needs: Sequence[str],
    will_write: Sequence[str],
    run: Callable[[], Any],
    background: bool = False,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Preview (confirm=False) or execute (confirm=True) a graintrace step.

    Parameters
    ----------
    tool:            tool name, for job labelling / messages.
    confirm:         gate flag from the caller. False -> preview only.
    resolved_params: the exact parameters that will be used (defaults filled).
    needs:           external dependency names (see deps._PROBES).
    will_write:      human-readable list of output paths/dirs this will create.
    run:             zero-arg callable that performs the actual work.
    background:      if True, execute via the job registry and return a job id.
    notes:           optional extra guidance shown in the preview.
    """
    dep_report = _dep_report(needs)
    if not confirm:
        return {
            "status": "preview",
            "tool": tool,
            "will_run": False,
            "resolved_params": resolved_params,
            "requires_external_tools": dep_report,
            "all_tools_ready": all(d["ok"] for d in dep_report),
            "will_write": list(will_write),
            "runs_in_background": background,
            "notes": notes,
            "next_step": (
                "Show these parameters to the user. If they approve, call this "
                "tool again with the same params and confirm=true."
            ),
        }

    # confirm=True: enforce dependencies before doing anything.
    missing_msg = deps.require(*needs) if needs else None
    if missing_msg:
        return {
            "status": "blocked",
            "tool": tool,
            "will_run": False,
            "message": missing_msg,
            "requires_external_tools": dep_report,
        }

    if background:
        job = jobs.submit(tool, run)
        return {
            "status": "started",
            "tool": tool,
            "job_id": job.id,
            "log_path": job.log_path,
            "message": (
                f"Started '{tool}' as background job {job.id}. "
                "Poll job_status(job_id) for progress."
            ),
        }

    result = run()
    return {"status": "done", "tool": tool, "result": result}
