import pandas as pd
import numpy as np
import os
from generate_random_crystal import CrystalGenerator

class SyntheticHEDMGenerator:
    def __init__(
            self,
            output_dir,
            ff_bounding_box,
            ff_strain_stdev,
            ff_grain_characteristics,
            nf_bounding_box,
            nf_dz,
            nf_spacing,
            random_seed=42,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.ff_bounding_box = np.array(ff_bounding_box, dtype=float).ravel()
        self.nf_bounding_box = np.array(nf_bounding_box, dtype=float).ravel()
        self.ff_grain_characteristics = ff_grain_characteristics
        self.ff_strain_stdev = float(ff_strain_stdev)
        self.random_seed = int(random_seed)
        self.dz_nf = float(nf_dz)
        self.a_nf = float(nf_spacing)

        self.ff_dir = os.path.join(self.output_dir, "FF")
        self.ff_neper_dir = os.path.join(self.ff_dir, "neper")
        self.nf_dir = os.path.join(self.output_dir, "NF")

        self._validate_init()

    def run(self, ff_iterations=10):
        self.generate_ff(iterations=ff_iterations)
        self.generate_nf()

        # Diagnostics
        print(f"NF bounding box is updated to: {self.nf_bounding_box}")

        # Count total NF points (same XY reused across layers)
        vertices_xy = self._build_nf_hex_vertex_lattice()
        z_layers = self._compute_nf_z_layers()

        print(f"NF lattice vertices per layer: {len(vertices_xy)}")
        print(f"NF number of layers: {len(z_layers)}")
        print(f"NF total points: {len(vertices_xy) * len(z_layers)}")
    
    ## FAR FIELD METHODS ------------------------------------------------
    def generate_ff(self, iterations=10):
        """
        Generates:
          - output_dir/FF/neper/voronoi.csv 
          - output_dir/FF/ff.csv            
        """
        os.makedirs(self.ff_neper_dir, exist_ok=True)

        np.random.seed(self.random_seed)

        base_csv = self._generate_ff_base(iterations=iterations)

        df = pd.read_csv(base_csv)
        df = self._append_elastic_strain(df)

        out_ff = os.path.join(self.ff_dir, "ff.csv")
        df.to_csv(out_ff, index=False)

        print(f"\nGenerated Far Field synthetic HEDM data at: {out_ff}\n")

        return out_ff

    def _generate_ff_base(self, iterations=10):
        """
        Uses CrystalGenerator to create voronoi tessellation + voronoi.csv.
        Writes everything under output_dir/FF/neper/.
        """

        cg = CrystalGenerator(
            output_dir=self.ff_neper_dir,
            bounding_box=self.ff_bounding_box,
            seed=self.random_seed,
            dim=3,
        )

        # Validate morphology via CrystalGenerator (single source of truth)
        try:
            cg.validate_morpho(self.ff_grain_characteristics)
        except Exception:
            cg.show_morpho_options(exit_after=True)

        cg.generate_tessellation(
            morpho_args=self.ff_grain_characteristics,
            iterations=int(iterations),
        )

        base_csv = os.path.join(self.ff_neper_dir, "voronoi.csv")
       
        return base_csv

    def _append_elastic_strain(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds symmetric elastic strain tensor columns in microstrain:
          eKen11..eKen33, with Gaussian(mean=0, stdev=self.ff_strain_stdev).
        """
        required = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Base FF CSV missing required columns: {missing}")

        n = len(df)
        stdev = self.ff_strain_stdev

        exx = np.random.normal(0.0, stdev, n)
        eyy = np.random.normal(0.0, stdev, n)
        ezz = np.random.normal(0.0, stdev, n)
        exy = np.random.normal(0.0, stdev, n)
        eyz = np.random.normal(0.0, stdev, n)
        exz = np.random.normal(0.0, stdev, n)

        # symmetric 3x3 flattened (row-major)
        df["eKen11"] = exx
        df["eKen12"] = exy
        df["eKen13"] = exz
        df["eKen21"] = exy
        df["eKen22"] = eyy
        df["eKen23"] = eyz
        df["eKen31"] = exz
        df["eKen32"] = eyz
        df["eKen33"] = ezz

        return df
    
    ## NEAR FIELD METHODS ------------------------------------------------
    def generate_nf(self):
        """
        Generates near-field layered CSVs:
          output_dir/NF/layer_000.csv
          output_dir/NF/layer_001.csv
          ...

        Each layer CSV columns:
          X, Y, Eul1, Eul2, Eul3
        """
        os.makedirs(self.nf_dir, exist_ok=True)

        tess_path = os.path.join(self.ff_neper_dir, "voronoi.tess")
        if not os.path.exists(tess_path):
            raise FileNotFoundError(
                f"Missing '{tess_path}'. Run FF first (use run() or generate_ff())."
            )

        np.random.seed(self.random_seed)

        seeds_xyz, seed_eulers = self._read_voronoi_tess_seeds_and_orientations(tess_path)
        vertices_xy = self._build_nf_hex_vertex_lattice()
        z_layers = self._compute_nf_z_layers()

        for k, z_layer in enumerate(z_layers):
            eulers_at_vertices = self._assign_eulers_for_layer(
                vertices_xy=vertices_xy,
                z_layer=z_layer,
                seeds_xyz=seeds_xyz,
                seed_eulers=seed_eulers,
            )
            self._write_nf_layer_csv(k, vertices_xy, eulers_at_vertices)

        print(f"\nGenerated Near Field synthetic HEDM data in folder: {self.nf_dir}\n")
        
        self._nf_visualize()
        print(f"NF lattice visualization saved in: {os.path.join(self.nf_dir, 'visualize')}\n")

        return self.nf_dir

    def _read_voronoi_tess_seeds_and_orientations(self, tess_path: str):

        with open(tess_path, "r") as f:
            lines = f.readlines()

        # Find **cell count
        ncell = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith("**cell"):
                # next non-empty line is ncell
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                ncell = int(lines[j].strip())
                break
        if ncell is None:
            raise ValueError("Failed to find '**cell' block / ncell in tess file.")

        # Find *seed block
        seed_idx = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith("*seed"):
                seed_idx = i
                break
        if seed_idx is None:
            raise ValueError("Failed to find '*seed' block in tess file.")

        seeds = np.zeros((ncell, 3), dtype=float)
        
        row = 0
        for i in range(seed_idx + 1, len(lines)):
            s = lines[i].strip()
            if not s:
                continue
            if s.startswith("*") or s.startswith("**"):
                break
            parts = s.split()
            # Expected: id x y z w
            if len(parts) < 5:
                continue
            sid = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            seeds[row, :] = (x, y, z)
            row += 1
            if row == ncell:
                break
        if row != ncell:
            raise ValueError(f"Parsed {row} seeds, expected {ncell}.")

        # Find *ori block
        ori_idx = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith("*ori"):
                ori_idx = i
                break
        if ori_idx is None:
            raise ValueError("Failed to find '*ori' block in tess file.")

        # Skip descriptor line (e.g., "euler-bunge:passive")
        j = ori_idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        # descriptor
        j += 1

        eulers = np.zeros((ncell, 3), dtype=float)
        row = 0
        for i in range(j, len(lines)):
            s = lines[i].strip()
            if not s:
                continue
            if s.startswith("*") or s.startswith("**"):
                break
            parts = s.split()
            if len(parts) < 3:
                continue
            eulers[row, :] = (float(parts[0]), float(parts[1]), float(parts[2]))
            row += 1
            if row == ncell:
                break
        if row != ncell:
            raise ValueError(f"Parsed {row} ori rows, expected {ncell}.")

        return seeds, eulers

    def _build_nf_hex_vertex_lattice(self):
        xmin, xmax, ymin, ymax, _, _ = self.nf_bounding_box
        a = self.a_nf
        rt3 = np.sqrt(3.0)

        # Flat-top hex center spacing
        dx = 1.5 * a              # center-to-center in x
        dy = rt3 * a              # center-to-center in y
        y_off = 0.5 * rt3 * a     # odd-column y offset

        # Vertex offsets around a flat-top hex center
        voff = np.array([
            [ a, 0.0],
            [ 0.5 * a,  0.5 * rt3 * a],
            [-0.5 * a,  0.5 * rt3 * a],
            [-a, 0.0],
            [-0.5 * a, -0.5 * rt3 * a],
            [ 0.5 * a, -0.5 * rt3 * a],
        ], dtype=float)

        # Make a generous grid of centers that covers bbox (include margin of 1 hex)
        i_min = int(np.floor((xmin - a) / dx)) - 2
        i_max = int(np.ceil((xmax + a) / dx)) + 2
        j_min = int(np.floor((ymin - a) / dy)) - 2
        j_max = int(np.ceil((ymax + a) / dy)) + 2

        verts = []
        for i in range(i_min, i_max + 1):
            cx = i * dx
            col_shift = y_off if (i % 2) else 0.0
            for j in range(j_min, j_max + 1):
                cy = j * dy + col_shift

                # generate vertices for this center
                v = voff + np.array([cx, cy])
                # clip vertices to bbox
                mask = (
                    (v[:, 0] >= xmin - 1e-9) & (v[:, 0] <= xmax + 1e-9) &
                    (v[:, 1] >= ymin - 1e-9) & (v[:, 1] <= ymax + 1e-9)
                )
                vv = v[mask]
                if len(vv):
                    verts.append(vv)

        if not verts:
            raise ValueError("NF hex vertex generation produced zero vertices. Check a_nf and nf_bounding_box.")

        verts = np.vstack(verts)

        # Deduplicate robustly
        key = np.round(verts, decimals=10)
        _, idx = np.unique(key, axis=0, return_index=True)
        verts = verts[np.sort(idx)]

        # Snap bbox to vertex extrema (so boundary is made of vertices)
        self.nf_bounding_box[0] = float(np.min(verts[:, 0]))
        self.nf_bounding_box[1] = float(np.max(verts[:, 0]))
        self.nf_bounding_box[2] = float(np.min(verts[:, 1]))
        self.nf_bounding_box[3] = float(np.max(verts[:, 1]))

        return verts

    def _compute_nf_z_layers(self):
        """
        Compute z layers using dz_nf and snap zmax_nf so that:
          zmax_nf = zmin_nf + (n_layers - 1) * dz_nf
        """
        zmin = float(self.nf_bounding_box[4])
        zmax = float(self.nf_bounding_box[5])
        dz = self.dz_nf
        if dz <= 0:
            raise ValueError("dz_nf must be > 0.")

        span = zmax - zmin
        if span < 0:
            raise ValueError("nf_bounding_box has zmax < zmin.")

        # At least one layer
        n_layers = int(np.floor(span / dz)) + 1
        if n_layers < 1:
            n_layers = 1

        zmax_snapped = zmin + (n_layers - 1) * dz
        self.nf_bounding_box[5] = zmax_snapped

        z_layers = zmin + dz * np.arange(n_layers, dtype=float)
        return z_layers

    def _assign_eulers_for_layer(self, vertices_xy, z_layer, seeds_xyz, seed_eulers, chunk_size=5000):
        """
        Assign Euler angles to each vertex using nearest Voronoi seed in 3D (Euclidean).
        """
        nverts = vertices_xy.shape[0]
        out = np.zeros((nverts, 3), dtype=float)

        # chunked brute-force to avoid huge (Nverts x Nseeds) memory
        for start in range(0, nverts, chunk_size):
            end = min(start + chunk_size, nverts)
            qxy = vertices_xy[start:end]
            q = np.column_stack([qxy[:, 0], qxy[:, 1], np.full(end - start, z_layer, dtype=float)])

            # distances squared: (Q,M,3) -> (Q,M)
            # Use broadcasting: q[:,None,:] - seeds[None,:,:]
            diff = q[:, None, :] - seeds_xyz[None, :, :]
            d2 = np.einsum("qmk,qmk->qm", diff, diff)
            idx = np.argmin(d2, axis=1)

            out[start:end, :] = seed_eulers[idx, :]

        return out

    def _write_nf_layer_csv(self, layer_idx, vertices_xy, eulers):
        """
        Write one NF layer file:
          output_dir/NF/layer_{layer_idx:03d}.csv

        Columns:
          X, Y, Eul1, Eul2, Eul3
        """
        out_path = os.path.join(self.nf_dir, f"layer_{layer_idx:03d}.csv")

        df = pd.DataFrame({
            "X": vertices_xy[:, 0],
            "Y": vertices_xy[:, 1],
            "Eul1": eulers[:, 0],
            "Eul2": eulers[:, 1],
            "Eul3": eulers[:, 2],
        })

        df.to_csv(out_path, index=False)
        return out_path

    def _nf_visualize(self):
        import matplotlib.pyplot as plt
        vis_dir = os.path.join(self.nf_dir, "visualize")
        os.makedirs(vis_dir, exist_ok=True)

        vertices_xy = self._build_nf_hex_vertex_lattice()
        z_layers = self._compute_nf_z_layers()

        xmin, xmax, ymin, ymax, zmin, zmax = map(float, self.nf_bounding_box)

        # Expand XY over all Z layers
        X = []
        Y = []
        Z = []
        for z in z_layers:
            X.append(vertices_xy[:, 0])
            Y.append(vertices_xy[:, 1])
            Z.append([z] * len(vertices_xy))

        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)

        fig, axs = plt.subplots(2, 1, figsize=(6, 10))

        # --- Top view (X,Y) ---
        axs[0].plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color='red',
        )
        axs[0].scatter(X, Y, s=5,color='black')
        
        axs[0].set_aspect("equal")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")

        # --- Side view (X,Z) ---
        axs[1].plot(
            [xmin, xmax, xmax, xmin, xmin],
            [zmin, zmin, zmax, zmax, zmin],
            color='red',
        )
        axs[1].scatter(X, Z, s=5,color='black')

        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Z")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "nf_lattice_overview.png"), dpi=300)
        plt.close(fig)

    def _validate_init(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if self.ff_bounding_box.size != 6:
            raise ValueError("ff_bounding_box must be [xmin,xmax,ymin,ymax,zmin,zmax].")
        if self.nf_bounding_box.size != 6:
            raise ValueError("nf_bounding_box must be [xmin,xmax,ymin,ymax,zmin,zmax].")

        fx0, fx1, fy0, fy1, fz0, fz1 = self.ff_bounding_box
        nx0, nx1, ny0, ny1, nz0, nz1 = self.nf_bounding_box
        if not (fx0 <= nx0 <= nx1 <= fx1 and fy0 <= ny0 <= ny1 <= fy1 and fz0 <= nz0 <= nz1 <= fz1):
            raise ValueError("nf_bounding_box must be fully enclosed within ff_bounding_box.")

        if self.ff_strain_stdev < 0:
            raise ValueError("ff_strain_stdev must be >= 0.")

        if self.dz_nf <= 0:
            raise ValueError("dz_nf must be > 0.")
        if self.a_nf <= 0:
            raise ValueError("a_nf must be > 0.")

        if not isinstance(self.ff_grain_characteristics, dict):
            raise TypeError("ff_grain_characteristics must be a dict (crystal_morpho_args).")