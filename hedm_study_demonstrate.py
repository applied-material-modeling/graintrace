from generate_random_crystal import CrystalGenerator
from scan_stitching_comparison import ScanStitchingComparison
from hedm_stitching_techniques.naive_stitching import NaiveStitching
from hedm_stitching_techniques.region_base_stitching import RegionBaseStitching
import os
import matplotlib.pyplot as plt
import sys

fsize = 14
plt.rcParams.update({
    "font.size": fsize,           # Global font size
    "axes.labelsize": fsize,      # Axis label size
    "axes.titlesize": fsize,      # Title size
    "xtick.labelsize": fsize,     # X tick label size
    "ytick.labelsize": fsize,     # Y tick label size
    "legend.fontsize": fsize,     # Legend font size
})

# INPUT -------------------------------------------------------
output_dir = "hedm_demonstrate"

# True crystal structure parameters
bounding_box = [-500, 500, -500, 500, -1000, 500]
# crystal_morpho_args = {"type": "gg", "mean": 80.0}
crystal_morpho_args = {"type": "diameq", 
                       "distribution": "lognormal",
                         "params": (130.0, 5.0)}

# HEDM scan parameters
nscan = 4
overlap_percentage = 25 # percentage units (0-100)

# Mimic experiment noise conditions
apply_noise=False
remove_minimum_volume=False

# gaussian distribution noise, 0.005 = 0.5%
noise_level=0.005
# grain volume minimum threshold
min_vol=0.0 

# Acceptable tolerance for comparison and stitching
position_tolerance = 10 # length units
orientation_tolerance = 0.5 # degrees
radius_tolerance = 0 # percentage units -- set to -1 to disable radius consideration

compare_position_tolerance = 50 # adjusted such that max_pos_error remains the same as this value increases
compare_orientation_tolerance = 5.0

### -------------------------------------------------------------
# uncomment the line below to see available morphology options
# CrystalGenerator.show_morpho_options(exit_after=True)
### -------------------------------------------------------------

seed_number = 42 # for reproducibility
test = False

# MAIN ----------------------------------------------------------
output_dir = os.path.abspath(output_dir)

# Generate true crystal structure
cg = CrystalGenerator(
    output_dir=output_dir,
    bounding_box=bounding_box,
    seed=seed_number,
)

# cg.generate_tessellation(
#     morpho_args=crystal_morpho_args,
# )

# # Simulate HEDM scans
cg.hedm_zscan(
    tess_file=output_dir + "/voronoi.tess",
    nstep=nscan,
    overlap_percentage=overlap_percentage,
    verbose=False,
    # Noise controls sections
    apply_noise=apply_noise,
    remove_minimum_volume=remove_minimum_volume,
    noise_level=noise_level,
    min_vol=min_vol,
)

# Perform stitching
# List of scan files
scan_files = [
    output_dir + f"/hedm_scan/scan_{i}.csv"
      for i in range(nscan)
]
stitch_output_csv = output_dir + "/huy_stitched.csv"


### TO DO: change stitching technique here -----------------------
# (demonstration of NaiveStitching)
# stitch = NaiveStitching(
#     scan_files=scan_files,
#     output_csv=stitch_output_csv,
# )
# stitch.run()


# Huy version of Stitching
weights = {
    "pos": 0.1,
    "ori": 1.0,
    "rad": 0,
}
min_neighbors = 5

zlo = bounding_box[4]
zhi = bounding_box[5]
overlap_fraction = overlap_percentage / 100.0

stitcher = RegionBaseStitching(
    scan_files=scan_files,
    output_csv=stitch_output_csv,
    position_tolerance=position_tolerance,
    orientation_tolerance=orientation_tolerance,
    radius_tolerance=radius_tolerance,
    weights=weights,
    min_neighbors=min_neighbors,
)

stitched = stitcher.run(zlo=zlo, zhi=zhi, overlap_fraction=overlap_fraction)

print("\nStitching complete.")
print(f"Stitched df shape: {stitched.df.shape}")
print(f"Output written to: {stitch_output_csv}\n")
### -------------------------------------------------------------


# Run comparison between true and stitched structures

weights = {
    "pos": 0.1,
    "ori": 1.0,
    "rad": 0,
}

compare = ScanStitchingComparison(
    output_dir=output_dir + "/comparison",
    true_csv=output_dir + "/voronoi.csv",
    stitch_csv=stitch_output_csv,
    position_tolerance=compare_position_tolerance,
    orientation_tolerance=compare_orientation_tolerance,
    radius_tolerance=radius_tolerance,
    weights=weights,
)
compare.run_comparison()
compare._get_unmatched_grains(
    bounding_box=bounding_box,
    pos_tol=compare_position_tolerance,
    ori_tol=compare_orientation_tolerance,
    plot=True,
)

### -------------------------------------------------------------










###
### Testing stuff
###

if test:
    CrystalGenerator.show_morpho_options()

    tests  = [
        {"type": "gg", "mean": 1.0},
        {"type": "lamellar", "n": 8, "v": "z"},
        {"type": "columnar", "n": 8, "v": "x"},
        {"type": "bamboo", "n": 8, "v": "y"},
        {"type": "diameq", "distribution": "lognormal", "params": (0.1, 0.03)},
        {"type": "size", "distribution": "weibull", "params": (2.5, 1.0)},
    ]

    cg = CrystalGenerator(
        output_dir=output_dir,
        bounding_box=bounding_box,
    )

    for m in tests:
        morpho_str = cg._build_morpho(m)
        print(f"Input: {m}")
        print(f"Corresponding morphology string: {morpho_str}\n")

    stitch = NaiveStitching(
        scan_files=[
            output_dir + "/hedm_scan/scan_0.csv",
            output_dir + "/hedm_scan/scan_1.csv",
            output_dir + "/hedm_scan/scan_2.csv",
        ],
        output_csv=output_dir + "/naive_stitched.csv",
    )
    stitch.run()

    compare = ScanStitchingComparison(
        output_dir=output_dir + "/comparison",
        true_csv=output_dir + "/voronoi.csv",
        stitch_csv=output_dir + "/voronoi.csv",
    )

    compare.run_comparison()
    compare._get_unmatched_grains(
        bounding_box=bounding_box,
        pos_tol=position_tolerance,
        ori_tol=orientation_tolerance,
        rad_tol=radius_tolerance,
        plot=True,
    )



