# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""graintrace: link grain-scale HEDM/EBSD data to CPFE simulations."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("graintrace")
except PackageNotFoundError:  # running from a source tree without install metadata
    __version__ = "0.1.0"

# Public name -> submodule that defines it. Imports are performed lazily (PEP 562)
# so that `import graintrace` does NOT eagerly pull heavy/optional third-party
# stacks (neml2, pyzag, torch, torch_geometric). Each symbol is imported only when
# first accessed, which keeps the top-level import cheap and lets consumers that
# don't need the compiled stack still `import graintrace`.
_LAZY_EXPORTS = {
    "BaseMaterialApproximationModel": "base_material_approximation",
    "ClusterAnalysisIndicator": "cluster_indicator",
    "NearFieldMeshBuilder": "construct_nf_mesh",
    "VoronoiMeshBuilder": "construct_voronoi_mesh",
    "VoxelMeshBuilder": "construct_voxel_mesh",
    "ExperimentResults": "experiment_postprocessing",
    "FieldFileNaming": "experiment_postprocessing",
    "CrystalGenerator": "generate_random_crystal",
    "GraphGrainMatcher": "grain_graph_matching",
    "GraphSpatialCluster": "graph_spatial_cluster",
    "IPFProcessor": "ipf_postprocess",
    "MaterialCalibration": "material_calibration",
    "NFGridConversion": "nf_grid_conversion",
    "plot_block_properties_distribution": "plot_postprocessing",
    "plot_macroscopic_stress_strain": "plot_postprocessing",
    "plot_block_properties_over_time": "plot_postprocessing",
    "plot_pole_figure": "plot_postprocessing",
    "plot_ipf_orientation_tracking": "plot_postprocessing",
    "plot_ipf_fragmentation_tracking": "plot_postprocessing",
    "OrientationTracker": "ipf_orientation_tracking",
    "FragmentationAnalyzer": "fragmentation",
    "IdentifyRareClusters": "rare_cluster_indicator",
    "select_smallest_cluster": "rare_criteria_selection_library",
    "select_highest_von_mises_from_components": "rare_criteria_selection_library",
    "select_highest_scalar": "rare_criteria_selection_library",
    "select_highest_norm_3x3_tensor": "rare_criteria_selection_library",
    "CPFESimulation": "run_cpfe_simulation",
    "ScanStitchingComparison": "scan_stitching_comparison",
    "SimilarityMetricLibrary": "similarity_metric_library",
    "SimulationResults": "simulation_postprocessing",
    "SyntheticHEDMGenerator": "synthetic_hedm_generator",
    "UniaxialTaylorModel": "taylor",
    "TaylorModel": "taylor",
    "NeperTessToGraphNN": "tess_to_gnn",
    "SimilarityMetric": "user_data_class",
    "WeightConfig": "user_data_class",
    "RareCriteria": "user_data_class",
}

__all__ = sorted(_LAZY_EXPORTS) + ["__version__"]

_INSTALL_DOCS = "https://applied-material-modeling.github.io/graintrace/install.html"

# Top-level distribution that provides each optional import, mapped to the
# extras group that installs it. Anything not listed here and not part of the
# PUMA-built compiled tier falls back to a plain `pip install <dist>` hint.
_EXTRA_FOR_DEPENDENCY = {
    "torch_geometric": "gnn",
    "mcp": "mcp",
    "meshio": "examples",
}

# Provided by the PUMA build, not by pip: there is no extras group to suggest.
_COMPILED_TIER_DEPENDENCIES = {"neml2"}


def _missing_dependency_message(name, module, missing):
    """Build an actionable message for a lazy import that hit a missing dependency.

    Args:
        name (str): the public graintrace symbol that was requested
        module (str): the graintrace submodule that defines it
        missing (str): the top-level third-party module that could not be imported

    Returns:
        str: a message naming the symbol, the missing dependency, the remedy,
        and the install documentation.
    """
    extra = _EXTRA_FOR_DEPENDENCY.get(missing)
    if extra is not None:
        remedy = f'install it with: pip install "graintrace[{extra}]"'
    elif missing in _COMPILED_TIER_DEPENDENCIES:
        remedy = (
            "it is part of the compiled tier and is not on PyPI: build it from "
            "PUMA's pinned submodule (install tier 2 or 3)"
        )
    else:
        remedy = f"install it with: pip install {missing}"
    return (
        f"graintrace.{name} is provided by graintrace.{module}, which requires "
        f"the optional dependency {missing!r}. {missing!r} is not installed; "
        f"{remedy}. See the graintrace install tiers at {_INSTALL_DOCS}."
    )


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        submodule = import_module(f".{module}", __name__)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        # A missing graintrace submodule is a packaging bug, not a missing
        # optional dependency; let it surface unchanged.
        if not missing or missing == __name__:
            raise
        raise ModuleNotFoundError(
            _missing_dependency_message(name, module, missing), name=missing
        ) from exc
    obj = getattr(submodule, name)
    globals()[name] = obj  # cache so subsequent lookups skip __getattr__
    return obj


def __dir__():
    return sorted(list(globals()) + list(_LAZY_EXPORTS))
