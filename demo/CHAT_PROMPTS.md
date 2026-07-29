# Live demo — Open WebUI / MCP chat script

Reproduce the full pipeline through the graintrace MCP tools (Claude Opus 4.8 +
the graintrace tool server). Each computing step previews first; you approve, and
it runs. Heavy steps return a `job_id` you poll with `job_status`. Paths below
assume the repo root is the MCP server's working directory; adjust as needed.

Run `python demo/generate_experiment.py` once first so `demo/experiment/` exists.

---

**0. Environment check**
> Check the graintrace dependency status and summarize what's available.

(expects: `gpu` ok, `neper` ok, `neml2-aoti` ok; `puma-opt` shows where your build is.)

**1. Inspect the raw data (Task 1 in action)**
> I have a far-field HEDM scan at `demo/experiment/hedm_scan/scan_0.csv`. Inspect it
> and tell me what you'd need from me before reconstructing.

(Claude calls `inspect_experiment` → reports columns, a suggested bounding box, a
unit guess, and that sample dimensions / loading / units must be confirmed.)

**2. Point it at the experiment metadata**
> Use `demo/experiment/sample.json` for all sample metadata from here on.

**3. Stitch the 4 scans**
> Stitch the 4 scans in `demo/experiment/hedm_scan/` using that sample.json.

(previews resolved params → you approve → `stitched.csv`.)

**4. Calibrate the material**
> Calibrate the crystal-plasticity parameters against
> `demo/experiment/strain-stress.csv` using the stitched grains for texture, on GPU.

(returns a job id; poll `job_status` → 6 calibrated parameters.)

**5. Reconstruct the microstructure + mesh**
> Reconstruct the FF microstructure from the stitched CSV (generate the GMSH mesh),
> using the sample.json bounding box.

(job id → `reconstruction.msh`, `orientations.dat`, the `ee` file.)

**6. Run CPFE (far field, GPU)**
> Run CPFE on the reconstruction using the calibrated material and the sample.json
> loading (uniaxial tension along z). Use cuda:0.

(previews the auto-built boundary + grid; approve → job id → poll to completion.)

**7. Find the rare events**
> Identify the rare high-Nye-tensor regions in the CPFE result and tell me which
> grains / locations to look at.

(job id → rare-cluster stats: centroids, severity, nearest grains.)

**8. Show me**
> Visualize the reconstruction grains, and the REI hotspots.

(Claude calls `visualize` → PNG paths under the workdir; open them, or `list_outputs`.)

---

Tips:
- If you skip step 2, steps 3/5/6 will return `needs_input` and ask for the sample
  dimensions / loading / units — that's the safety feature working.
- `list_jobs` / `job_log <id>` show background progress; `list_outputs` lists files.
- For interactive 3D, open `demo/out/FF/reconstruction.msh` or the REI VTK in ParaView.
