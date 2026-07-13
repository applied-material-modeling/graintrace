# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tool: generate synthetic FF+NF HEDM data for testing (SyntheticHEDMGenerator)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graintrace.mcp.app import mcp, workdir
from graintrace.mcp.confirm import gate


@mcp.tool()
def generate_synthetic_hedm(
    ff_bounding_box: List[float],
    ff_grain_characteristics: str,
    nf_bounding_box: List[float],
    nf_dz: float,
    nf_spacing: float,
    ff_strain_stdev: float = 1e-4,
    output_dir: Optional[str] = None,
    random_seed: int = 42,
    ff_iterations: int = 10,
    confirm: bool = False,
) -> dict:
    """Generate a synthetic microstructure with matched FF + NF HEDM data (via
    NEPER) for testing the pipeline end-to-end without real experiment files
    (wraps `SyntheticHEDMGenerator`).

    Parameters
    ----------
    ff_bounding_box / nf_bounding_box : [xlo,xhi,ylo,yhi,zlo,zhi] micrometers.
    ff_grain_characteristics : NEPER morpho string, e.g.
        'diameq:lognormal(130,5),aspratio(1,1,1)'.
    nf_dz, nf_spacing : NF layer thickness and in-plane point spacing (um).
    ff_strain_stdev : stdev of the synthetic elastic strain field.
    output_dir : output folder (defaults under the MCP workdir).
    ff_iterations : CVT iterations for the FF tessellation.

    Needs NEPER. Runs as a background job; writes FF/ and NF/ subfolders.
    """
    from graintrace.synthetic_hedm_generator import SyntheticHEDMGenerator

    if output_dir is None:
        output_dir = str(workdir() / "synthetic_hedm")
    resolved = {
        "output_dir": output_dir, "ff_bounding_box": ff_bounding_box,
        "ff_grain_characteristics": ff_grain_characteristics,
        "ff_strain_stdev": ff_strain_stdev, "nf_bounding_box": nf_bounding_box,
        "nf_dz": nf_dz, "nf_spacing": nf_spacing, "random_seed": random_seed,
        "ff_iterations": ff_iterations,
    }

    def _run():
        gen = SyntheticHEDMGenerator(
            output_dir=output_dir, ff_bounding_box=ff_bounding_box,
            ff_strain_stdev=ff_strain_stdev,
            ff_grain_characteristics=ff_grain_characteristics,
            nf_bounding_box=nf_bounding_box, nf_dz=nf_dz, nf_spacing=nf_spacing,
            random_seed=random_seed,
        )
        gen.run(ff_iterations=ff_iterations)
        return {"output_dir": output_dir, "ff_dir": f"{output_dir}/FF", "nf_dir": f"{output_dir}/NF"}

    return gate(
        tool="generate_synthetic_hedm", confirm=confirm, resolved_params=resolved,
        needs=["neper"], will_write=[output_dir], run=_run, background=True,
    )
