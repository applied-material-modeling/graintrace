---
segment: microstructure_generation
tool: generate_synthetic_hedm (morpho string) / CrystalGenerator.generate_tessellation
applies_to: NEPER tessellation of synthetic microstructures (equiaxed / lamellar / columnar / elongated)
defaults:
  seed: 42
  iterations_normal: 5000
  iterations_bimodal: 15000
  iterations_lognormal_tail_or_elongated: 20000
  reg: -reg 1 (skip for lamellar)
---

# Microstructure generation: recommended parameters

Based on 12 studied cases (equiaxed, lamellar, columnar, elongated × lognormal,
normal, bi-modal). These are the observed-best settings for generating a
synthetic microstructure with NEPER. In the MCP the morpho string feeds
`generate_synthetic_hedm`'s `ff_grain_characteristics`; the low-level API is
`graintrace.generate_random_crystal.CrystalGenerator.generate_tessellation()`.

## Usage

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

### Elongated / anisotropic grains (aspect ratio)

Append `aspratio(x, y, z)` to the morpho string. This matches
`examples/demonstrate_hedm_anisotropic.py`, which elongates ~3× along z. Use the
higher iteration budget (elongated recipes need the most relaxation), and keep
`-reg 1`:

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

Note: `aspratio` sets the grain *shape* only. Far-field HEDM cannot recover true
grain morphology from centroids (a per-scan FF tessellation is a ~⅓-radius-noisy
proxy); the genuine source of anisotropic morphology is NF-HEDM. Use the
elongated generator for benchmarks/ground truth, not to infer shape from FF.

## Required arguments

| Argument | Recommended value | Why |
|---|---|---|
| `seed` | `42` (or any fixed int) | Reproducibility: same seed ⇒ bit-identical `.tess`. |
| `morpho_args={"type": "raw", ...}` | pass `morpho_str` directly | Only path that exposes NEPER's full recipe grammar (mixtures, coupled constraints, columnar/lamellar, `aspratio`). |
| `extra_neper_args=["-reg", "1"]` | equiaxed / columnar / elongated | Removes sub-resolution artifact cells. **Skip for lamellar.** |
| `iterations` | 5000 (normal) · 15000 (bimodal) · 20000 (lognormal-tail / elongated) | More iterations reduce loss and improve agreement with the prescribed distribution. Beyond ~50,000, loss drops very slowly. |

What the API does:
1. Applies `-reg 1` cleanup + an optional post-hoc `r ≥ 5 µm` filter that catches
   the last ~1% of degenerate cells.
2. Writes canonical outputs (`voronoi.tess`, `.geo`, `.ori`, `.stcell`, `.csv`).
3. Derives `-n from_morpho` automatically when the morpho string carries a size
   scale (`diameq:...`); otherwise pass `n=<int>` explicitly.

## Known limitations (NEPER 4.10.2-45)

- **Lamellar takes a single scalar width only.** `lamellar(w=<distribution>)`
  inline SIGABRTs; `lamellar(w=file(...))` reads per-grain-id values from file.
  Varying-width lamellar therefore needs a multi-step build.
- **Columnar takes a single axis letter only.** `columnar(v=<axis_letter>)`
  (e.g. `v=z`) is the only working form; the vector form `columnar(v=(0,0,1))`
  **hangs NEPER indefinitely**.
- **Columnar cross-section σ has a Poisson floor.** Single-mode columnar hits the
  mean but delivers CV ≈ 0.36 regardless of the target σ. Bimodal columnar fits
  well (the mixture gives the optimizer more variance).
- **Bimodal `+`-mixture weights are by COUNT, not volume.** Two peaks at 40 and
  100 µm with 50/50 weights ⇒ 50/50 grain *counts*, not equal volume fractions.

## Verification (do this after generation)

Confirm the achieved grain-size distribution matches the prescribed one; plot
the histogram from `voronoi.stcell` against the target PDF. Disagreement means
too few iterations or one of the NEPER limitations above; adjusting iterations
and geometry (bounding-box size) often helps.

**Sample size:** use the *minimum* number of grains that faithfully represents
the target distribution; fewer grains ⇒ fewer elements ⇒ tractable CPFE. Add
grains only when the histogram or the CPFE quantity of interest hasn't converged.
