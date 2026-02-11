# DEPENDENCIES INSTALLATION GUIDES

These are to performed after downloading the python package.
`git clone git@github.com:applied-material-modeling/graintrace.git`

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

- check `environment.ymal` for full list of required Python packages with conda environment. This environment will be named `graintrace_env`. To create the same environment with conda, run:

```bash
conda env create -f environment.yml
conda activate graintrace_env
```

2. __NEPER with GMSH__

- Installation instruction from gmsh can be found at `https://gmsh.info/`. Downloading from source code is usually prefered.

- After installing gmsh, install NEPER via `https://neper.info/doc/introduction.html`. Often, a local `GSL` is required to install, as well as `OpenBLAS`. Gmsh installation is recommend before NEPER to avoid linking issues.

- Alternatively, the code `construct_voronoi_mesh.py` provide an automatic installations in its `check_dependencies` function. This only works on LINUX system and have been verified with `Ubuntu 20.04`. This code also should set the correct environment, pull the compatible `gmsh` and `NEPER` versions, as well as download and compile the relevant programs inside the system's home path directory via `home = os.path.expanduser("~")`. To set up automatic installation under this approach, create a new python file and run:

```python
from construct_voronoi_mesh import VoronoiMeshBuilder
builder = VoronoiMeshBuilder(
    input_csv="mwe_data/synthetic_data.csv",
    output_dir="synethic_out",
    bounding_box=[0,1,0,1,0,1],
)
```

3. __CUBIT/SCULPT__

- Coreform CUBIT (either National Labs, commerical or education licenses) are required. It will contain both CUBIT and SCULPT `https://coreform.com/`. The downloadables files, as well as the instructions, are ties with the account that have the license activated. After obtaining the CUBIT coreform license, log in with credentials and download the software.

4. __MOOSE with NEML2__

The required libraries can be obtained form the github packages below, however, it is recommended to follow the instructions below or the official websites to make sure the dependencies are satisfied:

- PUMA: A MOOSE app that runs CPFE `git@github.com:applied-material-modeling/puma.git` with branch: `development`
- MOOSE: `https://github.com/idaholab/moose.git` with branch: `master`
- NEML2: `git@github.com:applied-material-modeling/neml2.git` with bracnh: `main`

Here are the resources to successfully compile MOOSE with NEML2

- Built MOOSE from source: `https://mooseframework.inl.gov/getting_started/installation/hpc_install_moose.html`

- Built NEML2 from source: `https://applied-material-modeling.github.io/neml2/install.html` and `https://applied-material-modeling.github.io/neml2/tutorials-getting-started.html`

- Linking MOOSE and NEML2: `https://mooseframework.inl.gov/getting_started/installation/install_neml2.html` and `https://mooseframework.inl.gov/getting_started/installation/install_libtorch.html`

__Instructions__: at least worked for `Ubuntu 20.04` with the appropriate `mpi` and compiler packages. Check the above websites for prerequisites and dependencies.

- Here we assume the current path is in an empty folder. This folder will contain all of the MOOSE related programs. Also there is a current conda environment activated with the necessary dependencies. 

```bash
conda activate graintrace_env
mkdir projects
cd projects
```

- Build MOOSE: make sure the gcc / compilers are located in the correct path, usually it is `/usr/bin/mpicc`.

```bash
export CC=mpicc CXX=mpicxx FC=mpif90 F90=mpif90 F77=mpif77
git clone https://github.com/idaholab/moose.git
export MOOSE_DIR=${PWD}/moose
cd moose
git checkout master
export MOOSE_JOBS=12 METHODS=opt
cd scripts
./update_and_rebuild_petsc.sh
./update_and_rebuild_libmesh.sh
./update_and_rebuild_wasp.sh
cd ../../
```

- Obtain GPU-enable based libtorch (this is for CUDA 12.6 - find compatibility matrix at: `https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix`). If other versions are required, change the argument for the wget command from `https://pytorch.org/get-started/locally/`. Make sure to select `Stable` `Linux` `LibTorch`. Copy and paste the link in `Run this Command:`. Make sure to do this outside of moose and inside the `projects` folder.

```bash
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.10.0%2Bcu126.zip
unzip ibtorch-shared-with-deps-2.10.0%2Bcu126.zip
export LIBTORCH_DIR=${PWD}/libtorch
```

- Obtain the compatible NEML2 and compile NEML2 for MOOSE and python package.

```bash
cd moose
./configure --with-libtorch --with-neml2
./scripts/update_and_rebuild_neml2.sh
```

Once neml2 is compiled, some message like this will appear, run the `cd <messagaes>`.

```bash
****************************************************************************************************
NEML2 has been successfully installed.
To configure MOOSE with NEML2, run the following commands:
  cd <messages>
****************************************************************************************************
```

Look at the last line, if it said `config.status: framework/include/base/MooseConfig.h is unchanged`. Then the NEML2-LIBTORCH configurations point to the correct path.

- Compile PUMA with linked MOOSE-NEML2.

```bash
cd ../
git clone git@github.com:applied-material-modeling/puma.git
cd puma
git checkout origin/development
make -j 12
```

- Make sure the conda environment from `environment.ymal` is active. Then do:

```bash
./run_tests
```

If all tests passed, then it is successfully compiled.

- Finally, activate the python package of NEML2. The compatible NEML2 folder is inside MOOSE at `moose/framework/contrib/`

```bash
cd ../
cd moose/framework/contrib/neml2
pip install . -v
```

## Minimum working example

Run code `cpfe_nfff_demonstrate.py` to check if the required programs are installed correctly.

- Ensure that the environment variables for the programs are specified correctly. For `Coreform CUBIT`, Locate this line of codes and change the correct environment variables.

```python
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

- Ensure that the `moose_run_file` points to the correct executable for PUMA (if follows to previous instruction, this file should be located at `projects/puma/puma-opt`). Locate this line and change to the correct path.

```python
moose_run_file="/home/tranh/projects/aps_build/puma/puma-opt"
```

- Locate these lines and reduce the `ncore` and `device_batch` as needed. In addition, set `devide=cpu` if `cuda` fails.

```python
ncore = 24
device = "cuda:0"
device_batch = 1000
```

This code will:

- Generate synthetic data represents NF and FF HEDM
- Use NEPER/GMSH to reconstruct crystal structure from FF
- Use CUBIT/SCULPT/NEML2 to reconstruct crystal structure from NF
- Use MOOSE/NEML2 to run CPFE simulations

In the end, navigate to `minimum_eexample_cpfe/simulation/cpfe_run.log` to make sure the simulation is completed.

## Check other python packages

Run these scripts to check if all required python packages are installed. If the scripts can run to completion without errors, then the python packages can be considered properly installed and functional.

- `demonstrate_graintracking.py`

- `demonstrate_postprocess.py`

- `rei_demonstrate_example_2D.py`

- `hedm_study_demonstrate.py`
