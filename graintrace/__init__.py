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

from __future__ import annotations

from .base_material_approximation import BaseMaterialApproximationModel
from .cluster_indicator import ClusterAnalysisIndicator
from .construct_nf_mesh import NearFieldMeshBuilder
from .construct_voronoi_mesh import VoronoiMeshBuilder
from .construct_voxel_mesh import VoxelMeshBuilder
from .experiment_postprocessing import ExperimentResults, FieldFileNaming
from .generate_random_crystal import CrystalGenerator
from .grain_graph_matching import GraphGrainMatcher
from .graph_spatial_cluster import GraphSpatialCluster
from .ipf_postprocess import IPFProcessor
from .material_calibration import MaterialCalibration
from .nf_grid_conversion import NFGridConversion
from .plot_postprocessing import (
    plot_block_properties_distribution,
    plot_macroscopic_stress_strain,
    plot_block_properties_over_time,
    plot_pole_figure,
)
from .rare_cluster_indicator import IdentifyRareClusters
from .rare_criteria_selection_library import (
    select_smallest_cluster,
    select_highest_von_mises_from_components,
    select_highest_scalar,
    select_highest_norm_3x3_tensor,
)
from .run_cpfe_simulation import CPFESimulation
from .scan_stitching_comparison import ScanStitchingComparison
from .similarity_metric_library import SimilarityMetricLibrary
from .simulation_postprocessing import SimulationResults
from .synthetic_hedm_generator import SyntheticHEDMGenerator
from .taylor import UniaxialTaylorModel, TaylorModel
from .tess_to_gnn import NeperTessToGraphNN
from .user_data_class import SimilarityMetric, WeightConfig, RareCriteria
