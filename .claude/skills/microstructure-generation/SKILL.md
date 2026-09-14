---
name: microstructure-generation
description: >
  Vetted parameter recommendations for microstructure GENERATION with NEPER via
  CrystalGenerator (morpho strings, iteration counts, aspratio/lamellar/columnar
  recipes), distilled from 12 studied cases (equiaxed/lamellar/columnar/elongated
  × lognormal/normal/bimodal). Use when choosing a NEPER morpho recipe or iteration
  budget, and for the mandatory post-generation distribution check. Meshing lives in
  the separate `/meshing` skill. Same content ships as MCP recipe
  'microstructure_generation'.
---

# Microstructure generation (NEPER via CrystalGenerator)

These recommendations are based on observations from 12 different cases of
microstructures (equiaxed, lamellar, columnar and elongated) and grain size
distributions (lognormal, normal and bi-modal). For meshing the generated grid,
see the `/meshing` skill.

Env: `conda activate moose-src`. External tool: **NEPER**.
Example: `examples/demonstrate_hedm_anisotropic.py` (12-case anisotropic study);
`examples/demonstrate_synthetic_cpfe.py` also generates then meshes + runs CPFE.

---

**API:** `graintrace.generate_random_crystal.CrystalGenerator.generate_tessellation()`

**Usage:**

`seed` defaults to `None` → a fresh random microstructure each run (the drawn seed is printed so
you can reproduce it). Pass an explicit `seed` (as below) for a reproducible tessellation.

```python
from graintrace.generate_random_crystal import CrystalGenerator

cg = CrystalGenerator(
    output_dir="out",
    bounding_box=[-500, 500, -500, 500, -500, 500],   # 1 mm cube
    dim=3,
    seed=42,
)

# Equiaxed, normal size distribution
cg.generate_tessellation(
    morpho_args={"type": "raw", "morpho_str": "diameq:normal(100, 20)"},
    iterations=5000,
    extra_neper_args=["-reg", "1"],
)
```

**Elongated / anisotropic grains (aspect ratio).** Append `aspratio(x, y, z)` to
the morpho string. This matches `examples/demonstrate_hedm_anisotropic.py`, which
elongates ~3× along z. Use the higher iteration budget and keep `-reg 1`:

```python
z_aspect = 3.0
cg.generate_tessellation(
    morpho_args={
        "type": "raw",
        "morpho_str": f"diameq:lognormal(130, 5),aspratio(1, 1, {z_aspect})",
    },
    iterations=20000,            # elongated -> use the 20k budget
    extra_neper_args=["-reg", "1"],
)
```

`aspratio` sets grain *shape* only. Far-field HEDM cannot recover true grain
morphology from centroids (a per-scan FF tessellation is a ~⅓-radius-noisy
proxy); NF-HEDM is the genuine source of grain morphology. Use the elongated
generator for benchmarks/ground truth, not to infer shape from FF.

**Explanation of required arguments:**

| Argument | Recommended value | Why |
|---|---|---|
| `seed` | `42` (or any fixed int) | Reproducibility: same seed ⇒ bit-identical `.tess`. |
| `morpho_args={"type": "raw", ...}` | pass `morpho_str` directly | Only path that exposes NEPER's full recipe grammar (mixtures, coupled constraints, columnar/lamellar, `aspratio`). |
| `extra_neper_args=["-reg", "1"]` | equiaxed / columnar / elongated | Removes sub-resolution artifact cells. Skip for lamellar. |
| `iterations` | 5000 (normal) · 15000 (bimodal) · 20000 (lognormal-tail / elongated) | More iterations help reduce loss and improve agreement with the prescribed distribution. However, iterations > `50,000` reduce loss very slowly. |

**Explanation of the API:**

1. Applies the `-reg 1` cleanup + optional post-hoc `r ≥ 5 µm` filter that catches
   the last ~1% of degenerate cells.
2. Writes canonical outputs (`voronoi.tess`, `.geo`, `.ori`, `.stcell`, `.csv`).
3. `-n from_morpho` is derived automatically when the morpho string carries a size
   scale (`diameq:...`); otherwise pass `n=<int>` explicitly.

### Known limitations (NEPER 4.10.2-45)

- **Lamellar takes a single scalar width only.** `lamellar(w=<distribution>)`
  inline SIGABRTs; `lamellar(w=file(...))` reads values from file for respective
  grain ids. Generating varying-width lamellar requires multiple steps.
- **Columnar takes a single axis letter only.** `columnar(v=<axis_letter>)`
  (e.g. `v=z`) is the only working form; the vector form `columnar(v=(0,0,1))`
  **hangs NEPER indefinitely**.
- **Columnar cross-section σ has a Poisson floor.** For single-mode columnar
  recipes, NEPER hits the mean but delivers CV ≈ 0.36 regardless of the target σ.
  Bimodal columnar fits well because the mixture gives the optimizer more variance.
- **Bimodal `+`-mixture weights are by COUNT, not volume.** Two peaks at 40 and
  100 µm with 50/50 weights means 50/50 grain counts, not equal volume fractions.

---

## Verification (mandatory after generation)

Confirm that the achieved grain-size distribution matches the prescribed one: plot
the histogram from `voronoi.stcell` against the target PDF. Non-agreement means
either too few iterations or a NEPER limitation from the list above. Playing with
the iterations and geometry (bounding box size) might help.

**Sample size guidance:** always use the **minimum number of grains that
faithfully represents the target distribution**. Fewer grains → fewer mesh
elements → tractable CPFE wall-clock. Add grains only when the distribution
histogram or the CPFE quantity of interest doesn't converge.
