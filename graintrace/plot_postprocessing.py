from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
import neml2

# --- barycentric coordinates in 2D triangle ---


def plot_block_properties_distribution(
    results: Any,
    time: float,
    tensor_prefix: str,
    order: int,
    output_folder: str = "postprocess_out",
    bins: int = 50,
) -> int:
    """
    Plot distribution across grains at a given physical time.
    """

    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1, or 2")

    block = results._block_df
    if block is None:
        raise RuntimeError("Block dataframe not loaded.")

    tvals = results.time.to_numpy()
    block_id = int(np.argmin(np.abs(tvals - time)))

    data, comp_names = results.get_tensor_block(
        tensor_prefix,
        order,
        sample="id",
        block_id=block_id,
        return_comp_names=True,
    )

    nplots = 1 if order == 0 else data.shape[1]

    if nplots == 1:
        nrows, ncols = 1, 1
    elif nplots == 3:
        nrows, ncols = 3, 1
    elif nplots == 6:
        nrows, ncols = 3, 2
    elif nplots == 9:
        nrows, ncols = 3, 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False,
    )

    axes = axes.ravel()

    if order == 0:
        axes[0].hist(data, bins=bins, density=True)
        axes[0].set_title(f"{tensor_prefix}")
        axes[0].set_xlabel(tensor_prefix)
        # axes[0].set_ylabel("count")
    else:
        for i in range(nplots):
            axes[i].hist(data[:, i], bins=bins, density=True)
            axes[i].set_title(f"{comp_names[i]}")
            axes[i].set_xlabel(comp_names[i])
            # axes[i].set_ylabel("count")

    fig.tight_layout()

    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    filename = output_folder / f"{tensor_prefix}_pdf_timeindex{block_id}.png"

    fig.savefig(filename, dpi=300)
    plt.close(fig)

    return block_id


def plot_macroscopic_stress_strain(
    results,
    stress_tensor_prefix,
    strain_tensor_prefix,
    volume_prefix,
    output_folder="postprocess_out",
):

    grain_ids = results.grain_ids

    T = results.get_tensor_block(
        volume_prefix, 0, sample="time", grain_id=grain_ids[0]
    ).shape[0]
    denom = np.zeros(T - 1)
    num_stress = np.zeros((T - 1, 9))
    num_strain = np.zeros((T - 1, 9))

    for gid in grain_ids:
        vol = results.get_tensor_block(volume_prefix, 0, sample="time", grain_id=gid)[
            1:, 0
        ]
        sig = results.get_tensor_block(
            stress_tensor_prefix, 2, sample="time", grain_id=gid
        )[1:, :]
        eps = results.get_tensor_block(
            strain_tensor_prefix, 2, sample="time", grain_id=gid
        )[1:, :]

        denom += vol
        num_stress += sig * vol[:, None]
        num_strain += eps * vol[:, None]

    macro_stress = num_stress / denom[:, None]
    macro_strain = num_strain / denom[:, None]

    pick = [0, 1, 2, 4, 5, 8]
    labels = ["xx", "xy", "xz", "yy", "yz", "zz"]

    S = macro_stress[:, pick]
    E = macro_strain[:, pick]

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), squeeze=False)
    axes = axes.ravel()

    for i in range(6):
        ax = axes[i]
        ax.plot(E[:, i], S[:, i])
        ax.set_xlabel(f"{strain_tensor_prefix}_{labels[i]}")
        ax.set_ylabel(f"{stress_tensor_prefix}_{labels[i]}")

    fig.tight_layout()

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    outpath = (
        output_folder
        / f"macro_stress_strain_{stress_tensor_prefix}_vs_{strain_tensor_prefix}.png"
    )

    fig.savefig(outpath)
    plt.close(fig)

    # print to get Ezz at time steps xxx
    # for t in range(T-1):
    #   print(f"Time step {t+1}: Ezz = {macro_strain[t,8]:.6e}")

    return outpath


def plot_block_properties_over_time(
    results,
    tensor_prefix,
    order,
    grain_ids=None,
    output_folder="postprocess_out",
):
    """
    Plot block (per-grain) properties over time.
    """

    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1, or 2")

    if grain_ids is None:
        grain_ids = results.grain_ids

    # use first grain to determine component structure
    data0, comp_names = results.get_tensor_block(
        tensor_prefix,
        order,
        sample="time",
        grain_id=grain_ids[0],
        return_comp_names=True,
    )

    # drop first time step (initial zero state)
    t = results.time.iloc[1:].to_numpy()
    ncomp = data0.shape[1]

    if ncomp == 1:
        nrows, ncols = 1, 1
    elif ncomp == 3:
        nrows, ncols = 3, 1
    elif ncomp == 6:
        nrows, ncols = 3, 2
    elif ncomp == 9:
        nrows, ncols = 3, 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 2.8 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for gid in grain_ids:
        data = results.get_tensor_block(
            tensor_prefix,
            order,
            sample="time",
            grain_id=gid,
        )[1:, :]

        for i in range(ncomp):
            axes[i].plot(t, data[:, i], label=f"grain {gid}")

    for i in range(ncomp):
        axes[i].set_xlabel("time")
        axes[i].set_ylabel(comp_names[i])

    for j in range(ncomp, len(axes)):
        axes[j].axis("off")

    if len(grain_ids) <= 10:
        axes[0].legend(fontsize=8)

    fig.tight_layout()

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fname = output_folder / f"{tensor_prefix}_order{order}_over_time.png"
    fig.savefig(fname)
    plt.close(fig)

    return fname


def plot_pole_figure(
    results,
    time,
    tensor_prefix,
    direction=[1, 1, 1],
    crystal_symmetry="432",
    device="cpu",
    output_folder="postprocess_out",
    construct_odf=False,
    DeLaValleePoussinKernel_val=0.1,
    odf_limits=[0.0, 3.0],
    orientation_type="mrp",
    orientation_units="radians",
    odf_ncontour=12,
):
    import neml2
    import torch
    import neml2.tensors
    import neml2.postprocessing

    block = results._block_df
    if block is None:
        raise RuntimeError("Block dataframe not loaded.")

    tvals = results.time.to_numpy()
    block_id = int(np.argmin(np.abs(tvals - time)))

    pdirection = torch.tensor(direction, dtype=torch.double, device=device)

    data, comp_names = results.get_tensor_block(
        tensor_prefix,
        order=1,
        sample="id",
        block_id=block_id,
        return_comp_names=True,
    )

    if orientation_type != "mrp":
        data = torch.tensor(data, dtype=torch.double, device=device)
        orientations = neml2.tensors.Rot.fill_euler_angles(
            neml2.tensors.Vec(data), orientation_type, orientation_units
        )
    else:
        orientations = neml2.tensors.Rot(torch.tensor(data, dtype=torch.double))

    neml2.postprocessing.pretty_plot_inverse_pole_figure(
        orientations,
        pdirection,
        crystal_symmetry=crystal_symmetry,
        sample_symmetry=crystal_symmetry,
    )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fname = (
        output_folder
        / f"inversepolefigure_discrete_{direction[0]}{direction[1]}{direction[2]}_timeindex{block_id}.png"
    )
    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    plt.close()

    neml2.postprocessing.pretty_plot_pole_figure_points(
        orientations,
        pdirection,
        crystal_symmetry=crystal_symmetry,
    )

    fname = (
        output_folder
        / f"polefigure_discrete_{direction[0]}{direction[1]}{direction[2]}_timeindex{block_id}.png"
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

    if construct_odf:
        odf = neml2.postprocessing.odf.KDEODF(
            orientations,
            neml2.postprocessing.odf.DeLaValleePoussinKernel(
                torch.tensor(DeLaValleePoussinKernel_val)
            ),
        )
        odf.optimize_kernel(verbose=True)
        print(odf.kernel.h)
        neml2.postprocessing.pretty_plot_pole_figure_odf(
            odf,
            pdirection,
            crystal_symmetry=crystal_symmetry,
            limits=odf_limits,
            ncontour=odf_ncontour,
        )

        fname_odf = (
            output_folder
            / f"polefigure_odf_{direction[0]}{direction[1]}{direction[2]}_timeindex{block_id}.png"
        )
        plt.tight_layout()
        plt.savefig(fname_odf, dpi=300)
        plt.close()

        return fname, fname_odf

    return fname
