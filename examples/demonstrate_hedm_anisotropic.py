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

"""Anisotropic (elongated) HEDM stitching study: spherical vs tessellation extents.

RegionBaseStitching classifies each grain into a region from its z-extent relative
to the scan overlap band. By default that extent is the equivalent-sphere `z +/- r`,
which badly under-estimates the true z-reach of *elongated* grains. This script
generates a z-elongated microstructure (NEPER `aspratio`), simulates overlapping FF
z-scans, and stitches it twice: with the spherical extent and with the opt-in
NEPER-tessellation extent (`refine_extents=True`), then scores both against the
ground truth to show where the tessellation refinement actually helps.

This is a BENCHMARK HARNESS, not a demonstration of a guaranteed win. Whether the
tessellation helps is empirical, and in this elongated regime it often does NOT beat
the sphere, for two reasons worth understanding:

  1. Per-scan clipping: the stitcher tessellates each scan/accumulator over that scan's
     z-FOV, so an elongated grain's cell is clipped at the FOV boundary; the
     tessellation cannot see the grain's full z-extent either (it only flags that the
     grain reaches the boundary). It recovers the true extent only for grains that fit
     entirely inside one scan, and those are interior (CORE) grains whose region never
     mattered.
  2. Partial-view centroids: an elongated grain seen in two scans yields two different
     partial-volume centroids, which dominates the matching/merge error for both extent
     models (note the large mean_pos / mean_rad below).

So this script is here to *measure* the effect honestly. See CLAUDE.md "Region
classification" for the extent-model caveats (tessellation-from-centroids is a
~1/3-radius-noisy proxy; NF-HEDM is the real source of grain morphology).
"""
from graintrace.generate_random_crystal import CrystalGenerator
from graintrace.scan_stitching_comparison import ScanStitchingComparison
from graintrace.hedm_stitching_techniques.region_base_stitching import (
    RegionBaseStitching,
)
import os

# INPUT -------------------------------------------------------
output_dir = "hedm_anisotropic"

# True crystal structure: elongated ~3x along z via NEPER aspratio (use the 'raw'
# morpho type to pass any NEPER -morpho string verbatim).
bounding_box = [-500, 500, -500, 500, -1000, 500]
z_aspect = 3.0
crystal_morpho_args = {
    "type": "raw",
    "morpho_str": f"diameq:lognormal(130,5),aspratio(1,1,{z_aspect})",
}

# HEDM scan parameters
nscan = 4
overlap_percentage = 25  # percentage units (0-100)

# Noise off so the spherical-vs-tessellation difference is purely the extent model.
# (each knob applied only when > 0: position in length units, orientation in degrees,
#  radius as a relative fraction)
position_noise_std = 0.0
orientation_noise_std = 0.0
radius_noise = 0.0
noise_seed = 42
remove_minimum_volume = False
min_vol = 0.0

# Tolerances / weights
position_tolerance = 20  # length units
orientation_tolerance = 1  # degrees
radius_tolerance = -1  # -1 disables the radius gate
weights = {"pos": 0.1, "ori": 1.0, "rad": 0}
min_neighbors = 5

compare_position_tolerance = 20
compare_orientation_tolerance = 5.0

seed_number = 42

# MAIN ----------------------------------------------------------
output_dir = os.path.abspath(output_dir)

# 1) Generate the elongated ground-truth microstructure
cg = CrystalGenerator(
    output_dir=output_dir,
    bounding_box=bounding_box,
    seed=seed_number,
)
cg.generate_tessellation(morpho_args=crystal_morpho_args)

# 2) Simulate overlapping FF z-scans
cg.hedm_zscan(
    tess_file=output_dir + "/voronoi.tess",
    nstep=nscan,
    overlap_percentage=overlap_percentage,
    verbose=False,
    position_noise_std=position_noise_std,
    orientation_noise_std=orientation_noise_std,
    radius_noise=radius_noise,
    noise_seed=noise_seed,
    remove_minimum_volume=remove_minimum_volume,
    min_vol=min_vol,
)

scan_files = [output_dir + f"/hedm_scan/scan_{i}.csv" for i in range(nscan)]
zlo, zhi = bounding_box[4], bounding_box[5]
overlap_fraction = overlap_percentage / 100.0
true_csv = output_dir + "/voronoi.csv"


def stitch_and_score(tag, refine_extents):
    """Stitch with the chosen extent model and score against ground truth."""
    stitch_csv = f"{output_dir}/stitched_{tag}.csv"
    stitcher = RegionBaseStitching(
        scan_files=scan_files,
        output_csv=stitch_csv,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        radius_tolerance=radius_tolerance,
        weights=weights,
        min_neighbors=min_neighbors,
        refine_extents=refine_extents,  # False -> spherical z +/- r; True -> NEPER tessellation
        tess_weighted=True,
        update_centroid=False,  # keep FF centroids (see docs: cell centroid hurts matching)
        neper_env=cg.env,  # reuse the working NEPER env
    )
    stitcher.run(zlo=zlo, zhi=zhi, overlap_fraction=overlap_fraction)

    cmp = ScanStitchingComparison(
        output_dir=f"{output_dir}/comparison_{tag}",
        true_csv=true_csv,
        stitch_csv=stitch_csv,
        position_tolerance=compare_position_tolerance,
        orientation_tolerance=compare_orientation_tolerance,
        radius_tolerance=radius_tolerance,
        weights=weights,
    )
    cmp.run_comparison()
    return cmp.metrics


# 3) Stitch both ways and compare
print("\n=== Stitching elongated grains: spherical vs tessellation extents ===\n")
metrics = {
    "spherical": stitch_and_score("spherical", refine_extents=False),
    "tessellation": stitch_and_score("tessellation", refine_extents=True),
}

# 4) Report
hdr = f"{'mode':<14}{'matched':>9}{'splits':>8}{'merges':>8}{'mean_pos':>10}{'mean_ori':>11}"
print("\n" + hdr)
print("-" * len(hdr))
for tag, m in metrics.items():
    print(
        f"{tag:<14}{m['n_matched']:>9}{m['n_splits']:>8}{m['n_merges']:>8}"
        f"{m['mean_pos_abs_error']:>10.3f}{m['mean_ori_error']:>11.2e}"
    )

d_matched = metrics["tessellation"]["n_matched"] - metrics["spherical"]["n_matched"]
print(
    f"\nTessellation vs spherical: {d_matched:+d} matched grains (elongation ~{z_aspect:g}x in z).\n"
    "This is a measurement, not a guaranteed win. In practice the tessellation rarely beats\n"
    "the sphere here: per-scan tessellation clips an elongated grain's cell at the scan FOV\n"
    "(so it cannot recover the full z-extent either), and the partial-view centroids of\n"
    "elongated grains dominate the matching error (note the large mean_pos / mean_rad).\n"
    "The genuine source of anisotropic grain morphology is NF-HEDM, not an FF tessellation."
)
