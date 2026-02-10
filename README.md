# INSTALLATION GUIDES

## Required programs

- python >= 3.8.10
- conda with pip
- NEPER
- gmsh
- CUBIT/SCULPT
- MOOSE with PUMA app
- NEML2

## Installation instructions

1. __Required python packages__

- check `environment.ymal` for full list of required Python packages with conda environment. To create the same environment with conda, run:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

2. __NEPER with GMSH__

- Installation instruction from gmsh can be found at `https://gmsh.info/`. Downloading from source code is usually prefered.

- After installing gmsh, install NEPER via `https://neper.info/doc/introduction.html`. Often, a local `GSL` is required to install, as well as `OpenBLAS`. Gmsh installation is recommend before NEPER to avoid linking issues.

- Alternatively, the code `construct_voronoi_mesh.py` provide an automatic installations in its `check_dependencies` function. This only works on LINUX system and have been verified with `Ubuntu 20.04`. This code also should set the correct environment, pull the compatible `gmsh` and `NEPER` versions, as well as download and compile the relevant programs inside the system's home path directory via `home = os.path.expanduser("~")`. To set up automatic installation under this approach, create a new python file and run:

```bash
from construct_voronoi_mesh import VoronoiMeshBuilder
builder = VoronoiMeshBuilder(
    input_csv="testing_data/synthetic_data.csv",
    output_dir="synethic_out",
    bounding_box=[0,1,0,1,0,1],
)
```

3. __CUBIT/SCULPT__

- Coreform CUBIT (either National Labs, commerical or education licenses) are required. It will contain both CUBIT and SCULPT `https://coreform.com/`. The downloadables files, as well as the instructions, are ties with the account that have the license activated. After obtaining the CUBIT coreform license, log in with credentials and download the software.

4. __MOOSE with NEML2__

The required libraries can be obtained at, however, it is recommended to follow the instructions below or the official websites to make sure the dependencies are satisfied:

- PUMA: A MOOSE app that runs CPFE `git@github.com:applied-material-modeling/puma.git` with branch: `development`
- MOOSE: `https://github.com/idaholab/moose.git` with branch: `next`
- NEML2: `git@github.com:applied-material-modeling/neml2.git` with bracnh: `main`

Here are the resources to successfully compile MOOSE with NEML2

- Built MOOSE from source: `https://mooseframework.inl.gov/getting_started/installation/hpc_install_moose.html`

- Built NEML2 from source: `https://applied-material-modeling.github.io/neml2/install.html` and `https://applied-material-modeling.github.io/neml2/tutorials-getting-started.html`

- Linking MOOSE and NEML2: `https://mooseframework.inl.gov/getting_started/installation/install_neml2.html` and `https://mooseframework.inl.gov/getting_started/installation/install_libtorch.html`

These instructions below worked for `Ubuntu 20.04` with the appropriate `mpi` and compiler packages. Check the above websites for prerequisites and dependencies.



## Minimum working example

Run code `cpfe_nfff_demonstrate.py` to check if the required programs are installed correctly.

- Ensure that the environment variables for the programs are specified correctly. For `Coreform CUBIT`, Locate this line of codes and change the correct environment variables.

```bash
sculpt_config = {
    "mpirun": "/opt/Coreform-Cubit-2025.12/bin/mpi/bin/mpirun",
    "psculpt": "/opt/Coreform-Cubit-2025.12/bin/psculpt",
    "epu": "/opt/Coreform-Cubit-2025.12/bin/epu",
    "nprocs": int(ncore),
    "environment": {
        "OPAL_LIBDIR": "/opt/Coreform-Cubit-2025.12/bin/mpi/lib",
        "OPAL_PREFIX": "/opt/Coreform-Cubit-2025.12/bin/mpi",
    },
}
```

- Locate these lines and reduce the `ncore` and `device_batch` as needed. In addition, set `devide=cpu` if `cuda` fails.

```bash
ncore = 24
device = "cuda:0"
device_batch = 1000
```

This code will:

- Generate synthetic data represents NF and FF HEDM
- Use NEPER/GMSH to reconstruct crystal structure from FF
- Use CUBIT/SCULPT/NEML2 to reconstruct crystal structure from NF
- Use MOOSE/NEML2 to run CPFE simulations

## Check other python packages

Run these scripts to check if all required python packages are installed. If the scripts can run to completion without errors, then the python packages can be considered properly installed and functional.

- `demonstrate_graintracking.py`

- `demonstrate_postprocess.py`

- `rei_demonstrate_example_2D.py`

- `hedm_study_demonstrate.py`
