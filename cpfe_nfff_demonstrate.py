from synthetic_hedm_generator import SyntheticHEDMGenerator
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
output_dir = "cpfe_ff_nf_demonstrate"

ff_bounding_box = [-500, 500, -500, 500, -1000, 500]
ff_strain_stdev = 0.005

nf_bounding_box = [-200, 200, -200, 200, -200, 200]

crystal_morpho_args = {"type": "diameq", 
                       "distribution": "lognormal",
                         "params": (500.0, 5.0)}



## GENERATE SYNTHETIC FF + NF STRUCTURE ------------------------
synth_hedm_gen = SyntheticHEDMGenerator(
    output_dir=output_dir,
    ff_bounding_box=ff_bounding_box,
    ff_strain_stdev=ff_strain_stdev, 
    ff_grain_characteristics=crystal_morpho_args,
    nf_bounding_box=nf_bounding_box,
    nf_dz = 50,
    nf_spacing = 50,
    random_seed=42,
)

synth_hedm_gen.run(ff_iterations=10)