from generate_random_crystal import CrystalGenerator
from scan_stitching_comparison import ScanStitchingComparison
from hedm_stitching_techniques.naive_stitching import NaiveStitching
import os
import matplotlib.pyplot as plt

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
output_dir = "hedm_study/test"

# True crystal structure parameters
bounding_box = [-10, 10, -10, 10, -10, 10]
crystal_morpho_args = {"type": "gg", "mean": 3.0}


# HEDM scan parameters
nscan = 2
overlap_percentage = 0.0

# Mimic experiment noise conditions
apply_noise=False
remove_minimum_volume=False
noise_level=0.0 # multiplicative noise, with gaussian distribution
min_vol=0.0 # grain volume minimum threshold 

# Acceptable tolerance for comparison
position_tolerance = 0.1 # length units
orientation_tolerance = 0.1 # degrees
radius_tolerance = 0.1 # percentage units

### -------------------------------------------------------------
# uncomment the line below to see available morphology options
# CrystalGenerator.show_morpho_options()
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
cg.generate_tessellation(
    morpho_args=crystal_morpho_args,
)

# Simulate HEDM scans
cg.hedm_zscan(
    tess_file=output_dir + "/voronoi.tess",
    nstep=nscan,
    overlap_percentage=overlap_percentage,
    verbose=True,
    # Noise controls sections
    apply_noise=apply_noise,
    remove_minimum_volume=remove_minimum_volume,
    noise_level=noise_level,
    min_vol=min_vol,
)

# Perform stitching (here demonstrate naive stitching)
# List of scan files
scan_files = [
    output_dir + f"/hedm_scan/scan_{i}.csv" for i in range(nscan)
]
stitch_output_csv = output_dir + "/naive_stitched.csv"

stitch = NaiveStitching(
    scan_files=scan_files,
    output_csv=stitch_output_csv,
)
stitch.run()

# Run comparison between true and stitched structures
compare = ScanStitchingComparison(
    output_dir=output_dir + "/comparison",
    true_csv=output_dir + "/voronoi.csv",
    stitch_csv=stitch_output_csv,
)
compare.run_comparison()
compare._get_unmatched_grains(
    bounding_box=bounding_box,
    pos_tol=position_tolerance,
    ori_tol=orientation_tolerance,
    rad_tol=radius_tolerance,
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



