# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tool: calibrate crystal-plasticity parameters (MaterialCalibration + TaylorModel)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from graintrace.mcp.app import mcp, workdir
from graintrace.mcp.confirm import gate


def _cpfe_base() -> str:
    """Return the path to the packaged cpfe_base template directory."""
    # Lazy self-import to locate the package without a load-time cycle.
    import graintrace as _gt  # pylint: disable=import-outside-toplevel

    return str(Path(_gt.__file__).parent / "cpfe_base")


@mcp.tool()
def calibrate_material(
    data_dir: str,
    strain_stress_file: str,
    model_args: Optional[Dict[str, Any]] = None,
    data_args: Optional[Dict[str, Any]] = None,
    calibrate_args: Optional[Dict[str, Any]] = None,
    save_dir: Optional[str] = None,
    apply_elastic_correction: bool = False,
    strain_window: Optional[list] = None,
    confirm: bool = False,
) -> dict:
    """Calibrate 6 crystal-plasticity parameters (elastic E/nu/G, slip strength,
    Voce hardening slope/saturation) to a macro stress-strain curve + full-field
    elastic strains, using a NEML2 v3 + pyzag analytic-adjoint Taylor model with
    LBFGS (wraps `MaterialCalibration` with `TaylorModel`).

    Parameters
    ----------
    data_dir : folder of per-stress-level CSVs (O11..O33, coords, Eul0/1/2, eKen*).
    strain_stress_file : macro strain-stress CSV.
    model_args : overrides for TaylorModel(...) -- neml2_path (defaults to the
        packaged neml2_cpfe_calibration.i), npoints, nchunk, device
        ('cpu'|'cuda'), compile.
    data_args : overrides for load_experiment_data(...) -- npoints,
        full_field_strain_units, straintype ('eKen'|'eFab'), max_strain,
        n_grains, seed.
    calibrate_args : overrides for calibrate(...) -- maxiter, lr,
        max_iter_per_step, line_search_fn, plateau_rtol, plateau_window.
    save_dir : output folder (defaults under the MCP workdir).
    apply_elastic_correction / strain_window : optional elastic-slope correction.

    Needs NEML2 v3 + pyzag. Runs as a background job; writes
    calibrated_material.json.
    """
    # Lazy: heavy graintrace/neml2 submodules, imported only when the tool runs.
    # pylint: disable=import-outside-toplevel
    from graintrace.material_calibration import MaterialCalibration
    from graintrace.taylor import TaylorModel
    from graintrace.mcp import deps

    if save_dir is None:
        save_dir = str(workdir() / "material_calibration")

    # GPU policy: default to cuda when a GPU is available (caller can override).
    m_args = {
        "neml2_path": _cpfe_base() + "/neml2_cpfe_calibration.i",
        "npoints": 30,
        "nchunk": 2,
        "device": deps.default_device(),
        "compile": False,
        **(model_args or {}),
    }
    d_args = {
        "data_dir": data_dir,
        "strain_stress_file": strain_stress_file,
        "npoints": 30,
        "full_field_strain_units": "microstrain",
        "straintype": "eKen",
        "max_strain": 0.006,
        "n_grains": 100,
        "seed": 42,
        **(data_args or {}),
    }
    c_args = {
        "maxiter": 15,
        "lr": 0.3,
        "max_iter_per_step": 6,
        "line_search_fn": "strong_wolfe",
        "plateau_rtol": 1e-3,
        "plateau_window": 2,
        **(calibrate_args or {}),
    }
    resolved = {
        "model_args": m_args,
        "data_args": d_args,
        "calibrate_args": c_args,
        "save_dir": save_dir,
        "apply_elastic_correction": apply_elastic_correction,
        "strain_window": strain_window,
    }

    def _run():
        calib = MaterialCalibration(
            model_class=TaylorModel,
            model_args=m_args,
            data_args=d_args,
            save_dir=save_dir,
            apply_elastic_correction=apply_elastic_correction,
            strain_window=tuple(strain_window) if strain_window else None,
        )
        calib.calibrate(**c_args)
        return {
            "save_dir": save_dir,
            "calibrated_material": f"{save_dir}/calibrated_material.json",
        }

    return gate(
        tool="calibrate_material",
        confirm=confirm,
        resolved_params=resolved,
        needs=["neml2", "pyzag"],
        will_write=[save_dir],
        run=_run,
        background=True,
        notes="Iterative LBFGS fit; runs in the background.",
    )
