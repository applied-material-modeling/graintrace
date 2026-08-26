# graintrace benchmarks

Standalone **performance / scaling** scripts for the three heaviest graintrace
paths. They measure wall-clock timing / scaling on *your* machine.

Results are written to `benchmark/results/` (gitignored) and are **not** regression
gates — timings are machine-dependent. Each script **skips cleanly** (`SKIP: …`,
exit 0) when a dependency is missing.

## What each measures

| Script | Measures |
|---|---|
| `bench_graph_segmentation.py` | thread scaling of the graph-segmentation kernels: numba top-k prune (`n_jobs`) and NetworKit Leiden (`n_threads`) |
| `bench_cpfe.py` | CPFE wall time vs mesh **resolution** (element count) and **device_batch** on the GPU |
| `bench_calibration.py` | calibration wall time vs `device` (cpu/cuda), `n_grains`, `npoints` |

## Running

Run from the repo root.

```bash
# graph segmentation — pure Python, no external tools
python benchmark/bench_graph_segmentation.py --sizes 30 --jobs 1,2,4,8 --threads 1,2,4,8

# calibration — needs a working neml2 v3 + pyzag env; --probe checks it (exit 0/1)
python benchmark/bench_calibration.py --probe
python benchmark/bench_calibration.py --device auto --n-grains 50,100,250,500 --npoints 30

# CPFE — point --puma-bin at your built puma-opt (generates its own cube mesh)
python benchmark/bench_cpfe.py \
    --puma-bin /path/to/puma-opt \
    --resolution 16,24,32 --device-batch 5000,20000,50000

# CPFE output-frequency knobs (default cheap): --grid-transfer final|per_step|off,
# --exodus-output sync|per_step, --mesh-csv sync|per_step|off. Defaults avoid the
# per-step grid transfer, so a run-to-completion is much cheaper than the old default.

# One large run (e.g. 1M elements, ~3 loading steps, large batch), cheap output:
python benchmark/bench_cpfe.py --puma-bin /path/to/puma-opt \
    --resolution 100 --ncore 4 --device "cuda:0 cuda:1 cuda:2 cuda:3" \
    --device-batch 200000 --grid-transfer off --mesh-csv sync --exodus-output sync \
    --total-strain 0.03 --initialize-time 0.1 --dt 0.1 --total-time 0.4

# Aggregate a sweep's per-combo cpfe.csv (one dir per ncore/device_batch) into one table:
python benchmark/bench_cpfe.py --summarize /path/to/cpfe_sweep_<jobid>   # -> cpfe_summary.csv
```

Run each `bench_*.py --help` for the full flag list.

## Output

Each run creates `benchmark/results/<host>_<timestamp>/<name>/` with `<name>.csv`
and `<name>.json` (rows + host/GPU/git metadata). `bench_graph_segmentation.py
--plot` also writes a speedup PNG.
