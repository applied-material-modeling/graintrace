#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20          # = max ncore; srun -n <=16 subsets this (1 core/rank)
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --partition=gpuA100x4
#SBATCH --account=bibm-delta-gpu
#SBATCH --mem=240g
#SBATCH --job-name=graintrace_cpfe_sweep
#SBATCH --time=36:00:00
#SBATCH --output=graintrace_cpfe_sweep-%j.log
#SBATCH --mail-user=tranh@anl.gov
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail   # no -e: a failed (OOM/etc.) combo must not kill the whole sweep

# --- runtime environment ---------------------------------------------------
# cray-mpich must be loaded at RUN time so puma-opt's linked libmpi resolves.
module load PrgEnv-gnu gcc-native/14 cray-mpich craype-network-ofi cuda
export CC=gcc CXX=g++                 # AOTI/inductor needs a PLAIN host compiler

source ~/anaconda3/etc/profile.d/conda.sh
conda activate moose_neml2_v3

REPO=/projects/bibm/huydt2/moose_neml2_v3/graintrace
PUMA=/projects/bibm/huydt2/moose_neml2_v3/update_build/puma/puma-opt
export MOOSE_DIR=/projects/bibm/huydt2/moose_neml2_v3/update_build/puma/moose   # for R2IncrementToRate.py

# --- locate neml2 + torch runtime libs (libneml2_eager.so / libtorch) ------
NEML2_SO=$(find /projects/bibm/huydt2/moose_neml2_v3/update_build ~/anaconda3/envs/moose_neml2_v3 \
             -name libneml2_eager.so 2>/dev/null | head -1)
if [ -z "$NEML2_SO" ]; then echo "FATAL: libneml2_eager.so not found"; exit 1; fi
NEML2_LIB=$(dirname "$NEML2_SO")
TORCH_LIB=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__)+"/lib")')
export LD_LIBRARY_PATH="${NEML2_LIB}:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"   # for the batch-shell steps
echo "NEML2_LIB=$NEML2_LIB"
echo "TORCH_LIB=$TORCH_LIB"

# --- puma-opt wrapper: sets LD_LIBRARY_PATH INSIDE the srun task -----------
# (srun does not reliably forward LD_LIBRARY_PATH; the wrapper guarantees it.)
WRAP=/projects/bibm/huydt2/moose_neml2_v3/update_build/puma/puma-opt-wrap
cat > "$WRAP" <<EOF
#!/bin/bash
export LD_LIBRARY_PATH="${NEML2_LIB}:${TORCH_LIB}:\${LD_LIBRARY_PATH:-}"
exec ${PUMA} "\$@"
EOF
chmod +x "$WRAP"

# --- shim: graintrace launches `mpiexec`; on Cray the launcher is srun ------
mkdir -p "$SLURM_SUBMIT_DIR/.shim"
printf '#!/bin/bash\nexec srun --overlap --export=ALL "$@"\n' > "$SLURM_SUBMIT_DIR/.shim/mpiexec"
chmod +x "$SLURM_SUBMIT_DIR/.shim/mpiexec"
export PATH="$SLURM_SUBMIT_DIR/.shim:$PATH"

# --- sanity checks ---------------------------------------------------------
which mpiexec                          # -> the shim
nvidia-smi -L
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== preflight: does the wrapper resolve puma-opt's libs? ==="
srun --overlap --export=ALL -n 1 "$WRAP" --help >/dev/null 2>&1 \
  && echo "preflight OK: libs resolve inside the srun task" \
  || echo "preflight WARN: puma-opt --help failed (check libs / GPU in step)"

# --- sweep -----------------------------------------------------------------
cd "$REPO"
BASE=/scratch/bibm/huydt2/gt_bench/cpfe_sweep_${SLURM_JOB_ID}
RES=100                                   # res^3 = 1e6 elements -> distributed mesh
NCORE=20                                   # distributed mesh REQUIRES ncore>=2
DBS="25000"
for db in $DBS; do
  echo "===== ncore=20 device_batch=$db (distributed mesh) ====="
  python benchmark/bench_cpfe.py \
    --puma-bin "$WRAP" --moose-dir "$MOOSE_DIR" \
    --resolution "$RES" --device cuda:0 --ncore $NCORE \
    --device-batch "$db" \
    --distributed-mesh \
    --grid-transfer final --mesh-csv sync --exodus-output sync \
    --total-strain 0.03 --total-time 1.0 --dt 0.1 --initialize-time 0.01 \
    --out "${BASE}/ncore20_db${db}" --timeout 43000 \
    || echo "  FAILED db=$db -- continuing"
done
python benchmark/bench_cpfe.py --summarize "${BASE}"   # -> cpfe_summary.csv
