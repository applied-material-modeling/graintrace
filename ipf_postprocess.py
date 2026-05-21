import os
import shutil
from functools import reduce

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon

import neml2
import neml2.tensors
import neml2.postprocessing
from neml2 import tensors
from neml2.postprocessing import (
    symmetry_operators_as_R2,
    IPFReduction,
    StereographicProjection,
    LambertProjection,
)

import pyvista as pv


class IPFProcessor:
    available_projections = {
        "stereographic": StereographicProjection(),
        "lambert": LambertProjection(),
    }

    class IPFColorScheme:
        def __init__(
            self,
            v0=tensors.Vec(torch.tensor([0.0, 0.0, 1.0])),
            v1=tensors.Vec(torch.tensor([1.0, 0.0, 1.0])),
            v2=tensors.Vec(torch.tensor([1.0, 1.0, 1.0])),
        ):
            if not isinstance(v0, tensors.Vec):
                v0 = tensors.Vec(v0)
            if not isinstance(v1, tensors.Vec):
                v1 = tensors.Vec(v1)
            if not isinstance(v2, tensors.Vec):
                v2 = tensors.Vec(v2)

            v0 = v0 / v0.norm()
            v1 = v1 / v1.norm()
            v2 = v2 / v2.norm()

            self.v = [v0, v1, v2]

        def __call__(self, directions):
            if not isinstance(directions, tensors.Vec):
                directions = tensors.Vec(directions)

            pts = directions.torch()
            colors = torch.zeros(
                pts.shape[:-1] + (3,), dtype=pts.dtype, device=pts.device
            )

            for j in range(3):
                vj = self.v[j].torch().to(device=pts.device, dtype=pts.dtype)
                colors[..., j] = torch.sum(pts * vj, dim=-1)

            for i in range(3):
                vi = self.v[i].torch().to(device=pts.device, dtype=pts.dtype)
                mf = min(
                    [
                        torch.sum(
                            vi * vj.torch().to(device=pts.device, dtype=pts.dtype)
                        ).item()
                        for vj in self.v
                    ]
                )
                colors[..., i] = (colors[..., i] - mf) / (1.0 - mf)

            return torch.clamp(colors, 0.0, 1.0)

    def __init__(
        self,
        reduction=IPFReduction(),
        projection="stereographic",
        crystal_symmetry="1",
        sample_symmetry="1",
        save_dir=".",
    ):
        self.reduction = reduction
        self.projection = projection
        self.crystal_symmetry = crystal_symmetry
        self.sample_symmetry = sample_symmetry
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def _resolve_path(self, filename):
        if filename is None:
            return None
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.save_dir, filename)

    def _get_projection(self):
        return self.available_projections[self.projection]

    def _write_exodus_names_char_array(self, f, varname, names, maxlen=33):
        count = len(names)

        if "len_string" not in f.dimensions:
            f.createDimension("len_string", maxlen)

        if varname not in f.variables:
            f.createVariable(varname, "c", ("num_elem_var", "len_string"))

        arr = np.full((count, maxlen), b" ", dtype="S1")
        for i, name in enumerate(names):
            name_b = name.encode("ascii")[:maxlen]
            arr[i, : len(name_b)] = np.frombuffer(name_b, dtype="S1")

        f.variables[varname][:] = arr

    def ipf_color_chart(
        self,
        savefig_name="ipf_legend.png",
        axis_labels=("100", "110", "111"),
        ngrid=100,
        nline=100,
    ):
        colorizer = self.IPFColorScheme(
            self.reduction.v[0], self.reduction.v[1], self.reduction.v[2]
        )

        projection = self._get_projection()

        v0 = self.reduction.v[0].torch()
        v1 = self.reduction.v[1].torch()
        v2 = self.reduction.v[2].torch()

        tri2d = projection(torch.stack([v0, v1, v2], dim=0))
        xmin, xmax = tri2d[:, 0].min(), tri2d[:, 0].max()
        ymin, ymax = tri2d[:, 1].min(), tri2d[:, 1].max()

        xrang = torch.linspace(xmin, xmax, ngrid)
        yrang = torch.linspace(ymin, ymax, ngrid)
        X, Y = torch.meshgrid(xrang, yrang, indexing="xy")
        XY = torch.stack([X, Y], dim=-1)

        dirs = projection.inverse(XY)
        dirs[..., 2] = torch.abs(dirs[..., 2])

        rgb = colorizer(dirs)

        fig = plt.figure()
        ax = plt.subplot(111)
        im = ax.imshow(
            rgb.cpu().numpy(),
            extent=[
                xrang[0].item(),
                xrang[-1].item(),
                yrang[0].item(),
                yrang[-1].item(),
            ],
            origin="lower",
        )
        ax.axis("off")

        if axis_labels:
            plt.text(0.1, 0.11, axis_labels[0], transform=fig.transFigure)
            plt.text(0.86, 0.11, axis_labels[1], transform=fig.transFigure)
            plt.text(0.74, 0.88, axis_labels[2], transform=fig.transFigure)

        net_pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            vi = self.reduction.v[i].torch()
            vj = self.reduction.v[j].torch()
            fs = torch.linspace(0, 1, nline)
            pts3 = vi * (1.0 - fs).unsqueeze(-1) + vj * fs.unsqueeze(-1)
            pts3 /= torch.linalg.norm(pts3, dim=-1).unsqueeze(-1)
            pts2 = projection(pts3)
            net_pts.extend(pts2[:-1].cpu().numpy())

        poly = Polygon(net_pts, closed=True, edgecolor="k", fill=False)
        ax.add_patch(poly)
        im.set_clip_path(poly)

        savefig_path = self._resolve_path(savefig_name)
        plt.savefig(savefig_path, dpi=300)
        plt.close()

        return ax

    def get_reduced_ipf_directions(self, orientations, direction):
        if not isinstance(direction, tensors.Vec):
            direction = tensors.Vec(direction)
        direction = direction / direction.norm()

        if not isinstance(orientations, tensors.Rot):
            orientations = tensors.Rot(orientations)

        sample_symmetry_operators = symmetry_operators_as_R2(
            self.sample_symmetry, device=orientations.device
        )
        sample_directions = sample_symmetry_operators * direction

        crystal_directions = sample_directions.rotate(
            orientations.inv().dynamic.unsqueeze(-1)
        )

        symmetry_operators = symmetry_operators_as_R2(
            self.crystal_symmetry, device=orientations.device
        )

        equivalent_directions = (
            symmetry_operators * crystal_directions.dynamic.unsqueeze(-1)
        ).torch()

        cand = equivalent_directions.reshape(equivalent_directions.shape[0], -1, 3)

        normals = [
            n.torch().to(device=cand.device, dtype=cand.dtype) for n in self.reduction.n
        ]

        tol = 1e-9
        keep = reduce(
            torch.logical_and,
            [cand[..., 2] >= -tol]
            + [torch.sum(cand * n, dim=-1) >= -tol for n in normals],
        )

        nvalid = torch.sum(keep, dim=1)

        if torch.any(nvalid == 0):
            bad = torch.where(nvalid == 0)[0]
            raise RuntimeError(
                f"No valid IPF representative for orientations at indices {bad.tolist()}"
            )

        idx = torch.argmax(keep.to(torch.int64), dim=1)
        picked = cand[torch.arange(cand.shape[0], device=cand.device), idx]

        return tensors.Vec(picked)

    def get_ipf_color(self, orientations, direction):
        dirs = self.get_reduced_ipf_directions(orientations, direction)
        colorizer = self.IPFColorScheme(
            self.reduction.v[0], self.reduction.v[1], self.reduction.v[2]
        )
        return colorizer(dirs)

    def add_block_rgb_to_exodus(
        self,
        mesh_file,
        orientations_csv,
        direction,
        output_file="mesh_rgb.e",
        angle_convention="kocks",
        angle_type="radians",
    ):
        output_path = self._resolve_path(output_file)
        shutil.copyfile(mesh_file, output_path)

        ori = pd.read_csv(orientations_csv, header=None).to_numpy()
        ori = torch.tensor(ori)

        if angle_convention != "mrp":
            ori = tensors.Rot.fill_euler_angles(
                tensors.Vec(ori), angle_convention, angle_type
            )

        direction = torch.tensor(direction, dtype=torch.double)

        rgb = np.asarray(
            self.get_ipf_color(ori, direction=direction),
            dtype=np.float64,
        )

        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError("IPF color output must return shape (nblocks, 3)")

        with sio.netcdf_file(output_path, "a") as f:
            num_el_blk = int(f.dimensions["num_el_blk"])

            if rgb.shape[0] != num_el_blk:
                raise ValueError(
                    f"orientations rows ({rgb.shape[0]}) != num_el_blk ({num_el_blk})"
                )

            if "time_step" not in f.dimensions:
                f.createDimension("time_step", None)

            if "time_whole" not in f.variables:
                t = f.createVariable("time_whole", "f8", ("time_step",))
                t[0] = 0.0
            elif len(f.variables["time_whole"].data) == 0:
                f.variables["time_whole"][0] = 0.0

            if "num_elem_var" not in f.dimensions:
                f.createDimension("num_elem_var", 3)

            self._write_exodus_names_char_array(
                f,
                "name_elem_var",
                ["rgb_x", "rgb_y", "rgb_z"],
            )

            if "elem_var_tab" not in f.variables:
                tab = f.createVariable(
                    "elem_var_tab", "i4", ("num_el_blk", "num_elem_var")
                )
                tab[:] = np.ones((num_el_blk, 3), dtype=np.int32)

            for blk_idx in range(num_el_blk):
                n_elem = int(f.dimensions[f"num_el_in_blk{blk_idx + 1}"])
                rx, gy, bz = rgb[blk_idx]

                for var_idx, val in enumerate([rx, gy, bz], start=1):
                    vname = f"vals_elem_var{var_idx}eb{blk_idx + 1}"
                    if vname not in f.variables:
                        f.createVariable(
                            vname,
                            "f8",
                            ("time_step", f"num_el_in_blk{blk_idx + 1}"),
                        )
                    f.variables[vname][0, :] = np.full(n_elem, val, dtype=np.float64)

        return output_path

    def add_block_rgb_to_vtk(
        self,
        vtk_file,
        direction,
        output_file="mesh_rgb.vtk",
        angle_convention="kocks",
        angle_type="radians",
        orientation_fields=("Eul1", "Eul2", "Eul3"),
    ):
        output_path = self._resolve_path(output_file)

        mesh = pv.read(vtk_file)

        if len(orientation_fields) != 3:
            raise ValueError(
                "orientation_fields must contain exactly three field names"
            )

        for field in orientation_fields:
            if field not in mesh.cell_data:
                raise ValueError(f"Missing cell data field '{field}' in {vtk_file}")

        eul1 = np.asarray(mesh.cell_data[orientation_fields[0]])
        eul2 = np.asarray(mesh.cell_data[orientation_fields[1]])
        eul3 = np.asarray(mesh.cell_data[orientation_fields[2]])

        if not (len(eul1) == len(eul2) == len(eul3)):
            raise ValueError("Orientation cell data fields must have the same length")

        ori = np.column_stack((eul1, eul2, eul3))
        ori = torch.tensor(ori)

        if angle_convention != "mrp":
            ori = tensors.Rot.fill_euler_angles(
                tensors.Vec(ori), angle_convention, angle_type
            )

        direction = torch.tensor(direction, dtype=torch.double)

        rgb = np.asarray(
            self.get_ipf_color(ori, direction=direction),
            dtype=np.float64,
        )

        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError("IPF color output must return shape (ncells, 3)")

        if rgb.shape[0] != mesh.n_cells:
            raise ValueError(
                f"orientation rows ({rgb.shape[0]}) != number of cells ({mesh.n_cells})"
            )

        mesh.cell_data["rgb"] = rgb

        mesh.save(output_path)

        return output_path


"""
if __name__ == "__main__":
    processor = IPFProcessor(
        reduction=IPFReduction(),
        projection="stereographic",
        crystal_symmetry="432",
        sample_symmetry="432",
        save_dir="experiment_afrl/nf_reconstruction/mesh",
    )

    processor.ipf_color_chart(
        savefig_name="ipf_legend.png",
    )

    print("Generate ipf legends")

    out = processor.add_block_rgb_to_exodus(
        mesh_file="experiment_afrl/nf_reconstruction/mesh/mesh.e",
        orientations_csv="experiment_afrl/nf_reconstruction/mesh/orientations.csv",
        output_file="mesh_rgb.e",
        direction=[0.0, 0.0, 1.0],
        angle_convention="mrp",
    )
    print(f"Wrote {out}")

    vtk_out = processor.add_block_rgb_to_vtk(
        vtk_file="experiment_afrl/nf_reconstruction/mesh/merged_segmented_fixed_grid.vtk",
        output_file="merged_segmented_fixed_grid_rgb.vtk",
        direction=[0.0, 0.0, 1.0],
        angle_convention="bunge",
        angle_type="radians",
        orientation_fields=("Eul1", "Eul2", "Eul3"),
    )
    print(f"Wrote {vtk_out}")
"""
