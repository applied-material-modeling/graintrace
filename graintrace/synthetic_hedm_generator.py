# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import os
from .generate_random_crystal import CrystalGenerator


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

        self.z_layers = None
        self.vertices_xy = None

        self._validate_init()

    def run(self, ff_iterations: int = 10) -> None:
        self.generate_ff(iterations=ff_iterations)
        self.generate_nf()

        print(f"NF bounding box is updated to: {self.nf_bounding_box.tolist()}")

        # count total NF points (same XY reused across layers)
        vertices_xy = self._build_nf_hex_vertex_lattice()
        z_layers = self._compute_nf_z_layers()

        print(f"NF lattice vertices per layer: {len(vertices_xy)}")
        print(f"NF number of layers: {len(z_layers)}")
        print(f"NF total points: {len(vertices_xy) * len(z_layers)}")

    def generate_ff(self, iterations: int = 10) -> None:
        """Generate FF/neper/voronoi.csv and FF/ff.csv."""
        os.makedirs(self.ff_neper_dir, exist_ok=True)

        np.random.seed(self.random_seed)

        base_csv = self._generate_ff_base(iterations=iterations)

        df = pd.read_csv(base_csv)
        df = self._append_elastic_strain(df)

        out_ff = os.path.join(self.ff_dir, "ff.csv")
        df.to_csv(out_ff, index=False)

        print(f"\nGenerated Far Field synthetic HEDM data at: {out_ff}\n")

        return out_ff

    def _generate_ff_base(self, iterations: int = 10) -> pd.DataFrame:
        """Create voronoi tessellation + voronoi.csv under FF/neper/ via CrystalGenerator."""

        cg = CrystalGenerator(
            output_dir=self.ff_neper_dir,
            bounding_box=self.ff_bounding_box,
            seed=self.random_seed,
            dim=3,
        )

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
        """Add symmetric elastic strain columns eKen11..eKen33 (microstrain),
        Gaussian(0, self.ff_strain_stdev)."""
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

    def generate_nf(self) -> None:
        """Generate near-field layered CSVs NF/layer_XXX.csv (columns X, Y, Eul1, Eul2, Eul3)."""
        os.makedirs(self.nf_dir, exist_ok=True)

        tess_path = os.path.join(self.ff_neper_dir, "voronoi.tess")
        if not os.path.exists(tess_path):
            raise FileNotFoundError(
                f"Missing '{tess_path}'. Run FF first (use run() or generate_ff())."
            )

        np.random.seed(self.random_seed)

        seeds_xyz, seed_eulers = self._read_voronoi_tess_seeds_and_orientations(
            tess_path
        )
        self._build_nf_hex_vertex_lattice()
        self._compute_nf_z_layers()

        for k, z_layer in enumerate(self.z_layers):
            eulers_at_vertices = self._assign_eulers_for_layer(
                z_layer=z_layer,
                seeds_xyz=seeds_xyz,
                seed_eulers=seed_eulers,
            )
            self._write_nf_layer_csv(k, eulers_at_vertices)
            self._nf_visualize(
                plot_grid=False,
                plot_layer_property=True,
                layer_idx=k,
                eulers=eulers_at_vertices,
            )

        print(f"\nGenerated Near Field synthetic HEDM data in folder: {self.nf_dir}\n")

        self._nf_visualize()
        print(
            f"NF lattice visualization saved in: {os.path.join(self.nf_dir, 'visualize')}\n"
        )

        return self.nf_dir

    def _read_voronoi_tess_seeds_and_orientations(self, tess_path: str):

        with open(tess_path, "r") as f:
            lines = f.readlines()

        # find **cell count
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

        # find *seed block
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
            # expected: id x y z w
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

        # find *ori block
        ori_idx = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith("*ori"):
                ori_idx = i
                break
        if ori_idx is None:
            raise ValueError("Failed to find '*ori' block in tess file.")

        # skip descriptor line (e.g., "euler-bunge:passive")
        j = ori_idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
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

        # flat-top hex center spacing
        dx = 1.5 * a  # center-to-center in x
        dy = rt3 * a  # center-to-center in y
        y_off = 0.5 * rt3 * a  # odd-column y offset

        # vertex offsets around a flat-top hex center
        voff = np.array(
            [
                [a, 0.0],
                [0.5 * a, 0.5 * rt3 * a],
                [-0.5 * a, 0.5 * rt3 * a],
                [-a, 0.0],
                [-0.5 * a, -0.5 * rt3 * a],
                [0.5 * a, -0.5 * rt3 * a],
            ],
            dtype=float,
        )

        # grid of centers covering bbox with a 1-hex margin
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

                # vertices for this center, clipped to bbox
                v = voff + np.array([cx, cy])
                mask = (
                    (v[:, 0] >= xmin - 1e-9)
                    & (v[:, 0] <= xmax + 1e-9)
                    & (v[:, 1] >= ymin - 1e-9)
                    & (v[:, 1] <= ymax + 1e-9)
                )
                vv = v[mask]
                if len(vv):
                    verts.append(vv)

        if not verts:
            raise ValueError(
                "NF hex vertex generation produced zero vertices. Check a_nf and nf_bounding_box."
            )

        verts = np.vstack(verts)

        # deduplicate
        key = np.round(verts, decimals=10)
        _, idx = np.unique(key, axis=0, return_index=True)
        verts = verts[np.sort(idx)]

        # snap bbox to vertex extrema so boundary is made of vertices
        self.nf_bounding_box[0] = float(np.min(verts[:, 0]))
        self.nf_bounding_box[1] = float(np.max(verts[:, 0]))
        self.nf_bounding_box[2] = float(np.min(verts[:, 1]))
        self.nf_bounding_box[3] = float(np.max(verts[:, 1]))

        self.vertices_xy = verts

        return verts

    def _compute_nf_z_layers(self):
        """Compute z layers from dz_nf, snapping zmax to zmin + (n_layers-1)*dz_nf."""
        zmin = float(self.nf_bounding_box[4])
        zmax = float(self.nf_bounding_box[5])
        dz = self.dz_nf
        if dz <= 0:
            raise ValueError("dz_nf must be > 0.")

        span = zmax - zmin
        if span < 0:
            raise ValueError("nf_bounding_box has zmax < zmin.")

        # at least one layer
        n_layers = int(np.floor(span / dz)) + 1
        if n_layers < 1:
            n_layers = 1

        zmax_snapped = zmin + (n_layers - 1) * dz
        self.nf_bounding_box[5] = zmax_snapped

        z_layers = zmin + dz * np.arange(n_layers, dtype=float)

        self.z_layers = z_layers

        return z_layers

    def _assign_eulers_for_layer(
        self, z_layer, seeds_xyz, seed_eulers, chunk_size=5000
    ):
        """Assign Euler angles to each vertex from the nearest 3D Voronoi seed."""
        vertices_xy = self.vertices_xy

        nverts = vertices_xy.shape[0]
        out = np.zeros((nverts, 3), dtype=float)

        for start in range(0, nverts, chunk_size):
            end = min(start + chunk_size, nverts)
            qxy = vertices_xy[start:end]
            q = np.column_stack(
                [qxy[:, 0], qxy[:, 1], np.full(end - start, z_layer, dtype=float)]
            )

            diff = q[:, None, :] - seeds_xyz[None, :, :]
            d2 = np.einsum("qmk,qmk->qm", diff, diff)
            idx = np.argmin(d2, axis=1)

            out[start:end, :] = seed_eulers[idx, :]

        return out

    def _write_nf_layer_csv(
        self,
        layer_idx: int,
        eulers,
        *,
        tri_edge_size: float = 5.0,
        num_phases: int = 1,
        global_position: float = 0.0,
        nr_matches: float = 1.0,
        run_time: float = 0.1,
        updown: float = -1.0,
        confidence: float = 0.05,
        phase_nr: int = 1,
    ) -> str:

        out_path = os.path.join(self.nf_dir, f"layer_{layer_idx:03d}.csv")

        n = self.vertices_xy.shape[0]
        df = pd.DataFrame(
            {
                "%OrientationRowNr": np.arange(1, n + 1, dtype=float),
                "NrMatches": np.full(n, nr_matches, dtype=float),
                "RunTime": np.full(n, run_time, dtype=float),
                "X": self.vertices_xy[:, 0].astype(float),
                "Y": self.vertices_xy[:, 1].astype(float),
                "TriEdgeSize": np.full(n, tri_edge_size, dtype=float),
                "UpDown": np.full(n, updown, dtype=float),
                "Eul1": eulers[:, 0].astype(float),
                "Eul2": eulers[:, 1].astype(float),
                "Eul3": eulers[:, 2].astype(float),
                "Confidence": np.full(n, confidence, dtype=float),
                "PhaseNr": np.full(n, phase_nr, dtype=int),
            }
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"%TriEdgeSize {tri_edge_size:.6f}\n")
            f.write(f"%NumPhases {int(num_phases)}\n")
            f.write(f"%GlobalPosition {global_position:.6f}\n")
            df.to_csv(f, sep="\t", index=False)

        return out_path

    def _nf_visualize(
        self, plot_grid=True, plot_layer_property=False, layer_idx=0, eulers=None
    ):
        import matplotlib.pyplot as plt

        vis_dir = os.path.join(self.nf_dir, "visualize")
        os.makedirs(vis_dir, exist_ok=True)

        z_layers = self.z_layers
        vertices_xy = self.vertices_xy

        if plot_grid:

            xmin, xmax, ymin, ymax, zmin, zmax = map(float, self.nf_bounding_box)

            # expand XY over all Z layers
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

            # top view (X,Y)
            axs[0].plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                color="red",
            )
            axs[0].scatter(X, Y, s=10, color="black")

            axs[0].set_aspect("equal")
            axs[0].set_xlabel("X")
            axs[0].set_ylabel("Y")

            # side view (X,Z)
            axs[1].plot(
                [xmin, xmax, xmax, xmin, xmin],
                [zmin, zmin, zmax, zmax, zmin],
                color="red",
            )
            axs[1].scatter(X, Z, s=10, color="black")

            axs[1].set_xlabel("X")
            axs[1].set_ylabel("Z")

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "nf_lattice_overview.png"), dpi=300)
            plt.close(fig)

        if plot_layer_property:
            if eulers is None:
                raise ValueError("eulers must be provided for plot_layer_property.")

            fig, axs = plt.subplots(3, 1, figsize=(6, 12))
            titles = ["Eul1", "Eul2", "Eul3"]

            for i in range(3):
                sc = axs[i].scatter(
                    vertices_xy[:, 0],
                    vertices_xy[:, 1],
                    c=eulers[:, i],
                    s=40,
                    cmap="viridis",
                )
                axs[i].set_aspect("equal")
                axs[i].set_xlabel("X")
                axs[i].set_ylabel("Y")
                axs[i].set_title(f"Layer {layer_idx:03d} - {titles[i]}")

            fig.tight_layout()
            out = os.path.join(vis_dir, f"layer_{layer_idx:03d}_eulers.png")
            fig.savefig(out, dpi=300)
            plt.close(fig)

    def _validate_init(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        if self.ff_bounding_box.size != 6:
            raise ValueError("ff_bounding_box must be [xmin,xmax,ymin,ymax,zmin,zmax].")
        if self.nf_bounding_box.size != 6:
            raise ValueError("nf_bounding_box must be [xmin,xmax,ymin,ymax,zmin,zmax].")

        fx0, fx1, fy0, fy1, fz0, fz1 = self.ff_bounding_box
        nx0, nx1, ny0, ny1, nz0, nz1 = self.nf_bounding_box
        if not (
            fx0 <= nx0 <= nx1 <= fx1
            and fy0 <= ny0 <= ny1 <= fy1
            and fz0 <= nz0 <= nz1 <= fz1
        ):
            raise ValueError(
                "nf_bounding_box must be fully enclosed within ff_bounding_box."
            )

        if self.ff_strain_stdev < 0:
            raise ValueError("ff_strain_stdev must be >= 0.")

        if self.dz_nf <= 0:
            raise ValueError("dz_nf must be > 0.")
        if self.a_nf <= 0:
            raise ValueError("a_nf must be > 0.")

        if not isinstance(self.ff_grain_characteristics, dict):
            raise TypeError(
                "ff_grain_characteristics must be a dict (crystal_morpho_args)."
            )
