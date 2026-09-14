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

"""Plotting helpers for CPFE field distributions, stress-strain, and pole figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt


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

    # pylint: disable-next=protected-access  # internal results cache
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
    else:
        raise ValueError(f"Unsupported number of subplots: {nplots}")

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
    else:
        for i in range(nplots):
            axes[i].hist(data[:, i], bins=bins, density=True)
            axes[i].set_title(f"{comp_names[i]}")
            axes[i].set_xlabel(comp_names[i])

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
    """Plot volume-weighted macroscopic stress-strain curves per component."""
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

    data0, comp_names = results.get_tensor_block(
        tensor_prefix,
        order,
        sample="time",
        grain_id=grain_ids[0],
        return_comp_names=True,
    )

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
    else:
        raise ValueError(f"Unsupported number of components: {ncomp}")

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
):  # pylint: disable=dangerous-default-value  # read-only list defaults
    """Plot discrete inverse/direct pole figures (and optional ODF) at a time."""
    # pylint: disable=import-outside-toplevel  # torch/neml2 are heavy optional deps
    import torch
    from neml2 import types as _t
    from neml2 import texture as _texture

    from .orientation_helper import euler_to_matrix, mrp_to_matrix

    # pylint: disable-next=protected-access  # internal results cache
    block = results._block_df
    if block is None:
        raise RuntimeError("Block dataframe not loaded.")

    tvals = results.time.to_numpy()
    block_id = int(np.argmin(np.abs(tvals - time)))

    pdirection = torch.tensor(direction, dtype=torch.double, device=device)

    data, _comp_names = results.get_tensor_block(
        tensor_prefix,
        order=1,
        sample="id",
        block_id=block_id,
        return_comp_names=True,
    )

    # orientation_type="mrp" means neml2 v3 MRP (tan(theta/4)), matching the CPFE
    # `ori_rodrigues` output; mrp_to_matrix decodes it (not classical Rodrigues).
    if orientation_type != "mrp":
        R = euler_to_matrix(data, orientation_type, orientation_units)
    else:
        R = mrp_to_matrix(torch.as_tensor(data, dtype=torch.double))
    R = R.to(device=device, dtype=torch.double).contiguous()
    orientations = _t.MRP.from_matrix(_t.R2(R, 0))

    _texture.pretty_plot_inverse_pole_figure(
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

    _texture.pretty_plot_pole_figure_points(
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
        odf = _texture.odf.KDEODF(
            orientations,
            _texture.odf.DeLaValleePoussinKernel(
                torch.tensor(DeLaValleePoussinKernel_val)
            ),
        )
        odf.optimize_kernel(verbose=True)
        print(odf.kernel.h)
        _texture.pretty_plot_pole_figure_odf(
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


def plot_ipf_orientation_tracking(
    tracks,
    angle_convention,
    angle_type="radians",
    direction=(0.0, 0.0, 1.0),
    crystal_symmetry="432",
    sample_symmetry="1",
    projection="stereographic",
    arrow_color="black",
    start_size=45.0,
    step_size=3.0,
    line_width=1.0,
    arrow_scale=11.0,
    line_style="-",
    show_arrow=True,
    start_color=None,
    label_fontsize=None,
    output_folder="postprocess_out",
    savefig_name="ipf_orientation_tracking.png",
    axis_labels=("100", "110", "111"),
    ax=None,
):
    """Plot orientation trajectories ("points moving") in the IPF triangle.

    One IPF triangle (plain outline); each track is an IPF-colored start dot, a line
    through its per-step orientations (small step markers), and an arrowhead at the
    final step. Source-agnostic: the tracks may come from simulation or experiment.

    Args:
        tracks: sequence of ``(k_i, 3)`` orientation arrays, one per entity (neml2
            MRP when ``angle_convention="mrp"``, else Euler angles).
        angle_convention: REQUIRED (no default) so the caller must state how the
            tracks are encoded -- ``"mrp"`` (neml2 MRP, e.g. simulation ``ori_rodrigues``
            tracks) or an Euler convention ``"bunge"``/``"kocks"``/``"roe"`` (e.g.
            experiment tracks). A wrong value silently mis-projects, so it is explicit.
        angle_type: ``"radians"`` or ``"degrees"`` (Euler conventions only).
        direction: sample direction projected into the IPF (default z ``[0,0,1]``).
        crystal_symmetry: orbifold crystal symmetry string (e.g. ``"432"``).
        sample_symmetry: orbifold sample symmetry string.
        projection: ``"stereographic"`` or ``"lambert"``.
        arrow_color: color of the trajectory arrows. Either a single matplotlib
            color (applied to all tracks), a per-track sequence of colors, or the
            sentinel ``"ipf_final"`` -- each track is tinted by the IPF color of its
            final-stage orientation (so color reads as "where it ended up", which is
            redundant with position and needs no legend).
        start_size: size of the start dot.
        step_size: size of the intermediate load-step markers along each line.
        line_width: width of the trajectory lines and arrow shafts.
        arrow_scale: matplotlib ``mutation_scale`` for the arrowheads (head size).
        line_style: matplotlib linestyle for the trajectory lines/arrows. Either
            a single style (applied to all tracks) or a per-track sequence -- e.g.
            to distinguish which fragment each element belongs to.
        show_arrow: draw a final-step arrowhead on each track; ``False`` draws a
            plain line (no head) to declutter a crowded panel.
        start_color: color for every start dot; ``None`` uses each track's IPF
            color of its initial orientation. Pass ``"black"`` for a neutral origin.
        label_fontsize: font size for the triangle corner labels.
        output_folder: directory for the saved figure.
        savefig_name: output image filename.
        axis_labels: labels for the triangle corners.
        ax: existing axes to draw into; a new figure is created when ``None``.

    Returns:
        pathlib.Path: the saved figure path.
    """
    # pylint: disable=import-outside-toplevel  # heavy/optional deps (torch, neml2)
    import torch

    from .ipf_postprocess import IPFProcessor

    output_folder = Path(output_folder)
    created_fig = ax is None
    # Only touch the output folder when we actually save (i.e. no external ax);
    # when drawing into a supplied ax we return it and never write a file.
    if created_fig:
        output_folder.mkdir(parents=True, exist_ok=True)

    ipf = IPFProcessor(
        crystal_symmetry=crystal_symmetry,
        sample_symmetry=sample_symmetry,
        projection=projection,
        save_dir=str(output_folder) if created_fig else ".",
    )
    if ax is None:
        _, ax = plt.subplots()
    ipf.ipf_color_chart(
        savefig_name=None,
        axis_labels=axis_labels,
        fill=False,
        ax=ax,
        label_fontsize=label_fontsize,
    )

    direction_t = torch.tensor(list(direction), dtype=torch.float64)
    ipf_final = isinstance(arrow_color, str) and arrow_color == "ipf_final"
    colors = None if ipf_final else _resolve_arrow_colors(arrow_color, len(tracks))
    styles = _resolve_line_styles(line_style, len(tracks))
    for i, track in enumerate(tracks):
        proj = _project_ipf_track(ipf, track, angle_convention, angle_type, direction_t)
        if proj is None:
            continue
        pts, rgb0, rgb_final = proj
        line_color = rgb_final if ipf_final else colors[i]
        dot_color = rgb0 if start_color is None else start_color
        _draw_ipf_track(
            ax,
            pts,
            dot_color,
            line_color,
            start_size,
            step_size,
            draw_start=True,
            line_width=line_width,
            arrow_scale=arrow_scale,
            line_style=styles[i],
            show_arrow=show_arrow,
        )

    if not created_fig:
        return ax
    fname = output_folder / savefig_name
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    return fname


def _resolve_arrow_colors(arrow_color, n):
    """Normalize ``arrow_color`` to a list of ``n`` colors (one per entity).

    Accepts a single matplotlib color (a name string, or an RGB(A) tuple/list of
    3-4 floats) which is repeated, or a per-entity sequence of length ``n``.
    """
    if isinstance(arrow_color, str):
        return [arrow_color] * n
    seq = list(arrow_color)
    if not seq:
        return ["black"] * n
    if len(seq) in (3, 4) and all(isinstance(c, (int, float)) for c in seq):
        return [tuple(seq)] * n  # a single RGB(A) tuple
    if len(seq) != n:
        raise ValueError(
            f"arrow_color sequence length {len(seq)} != number of entities {n}"
        )
    return seq


def _resolve_line_styles(line_style, n):
    """Normalize ``line_style`` to a list of ``n`` matplotlib linestyles.

    Accepts a single linestyle (a string like ``"-"``/``"--"`` or an
    ``(offset, dashes)`` tuple) which is repeated, or a per-entity sequence of
    length ``n``.
    """
    if line_style is None:
        return ["-"] * n
    if isinstance(line_style, str):
        return [line_style] * n
    seq = list(line_style)
    if not seq:
        return ["-"] * n
    # a single (offset, on-off-seq) dash tuple, e.g. (0, (3, 1))
    if (
        len(seq) == 2
        and isinstance(seq[0], (int, float))
        and (seq[1] is None or isinstance(seq[1], (tuple, list)))
    ):
        return [tuple(seq)] * n
    if len(seq) != n:
        raise ValueError(
            f"line_style sequence length {len(seq)} != number of entities {n}"
        )
    return seq


def _project_ipf_track(ipf, track, angle_convention, angle_type, direction_t):
    """Project one ``(k, 3)`` orientation track to IPF points + its end colors.

    Returns ``(pts, rgb0, rgb_final)`` -- the IPF points, the IPF color of the
    first orientation, and the IPF color of the last (final-stage) orientation --
    or ``None`` for an empty/ill-shaped track.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    from .orientation_helper import euler_to_matrix, mrp_to_matrix

    arr = np.asarray(track, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return None
    t = torch.tensor(arr, dtype=torch.float64)
    rmat = (
        mrp_to_matrix(t)
        if angle_convention == "mrp"
        else euler_to_matrix(t, angle_convention, angle_type)
    )
    pts = np.asarray(ipf.get_ipf_points(rmat, direction_t), dtype=float)
    rgb0 = np.asarray(ipf.get_ipf_color(rmat[:1], direction_t), dtype=float)[0]
    rgb_final = np.asarray(ipf.get_ipf_color(rmat[-1:], direction_t), dtype=float)[0]
    return pts, rgb0, rgb_final


def _draw_ipf_track(
    ax,
    pts,
    rgb0,
    arrow_color,
    start_size,
    step_size,
    draw_start=True,
    line_width=1.0,
    arrow_scale=11.0,
    line_style="-",
    show_arrow=True,
):
    """Draw a single IPF trajectory: line, final-segment arrowhead, step markers.

    The line runs to the second-to-last point and the final segment is drawn as
    the arrow (head tip exactly at the last point -> no overshoot past the head).
    ``line_style`` is a matplotlib linestyle applied to both the line and the
    arrow shaft (e.g. distinguish fragments by ``"-"``/``"--"``/``"-."``/``":"``).
    When ``show_arrow`` is ``False`` the whole path is a plain line (no arrowhead),
    which declutters a crowded panel.
    """
    if len(pts) >= 2:
        if show_arrow:
            ax.plot(
                pts[:-1, 0],
                pts[:-1, 1],
                color=arrow_color,
                lw=line_width,
                ls=line_style,
                zorder=2,
            )
            ax.annotate(
                "",
                xy=(pts[-1, 0], pts[-1, 1]),
                xytext=(pts[-2, 0], pts[-2, 1]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": arrow_color,
                    "lw": line_width,
                    "linestyle": line_style,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": arrow_scale,
                },
                zorder=2,
            )
        else:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=arrow_color,
                lw=line_width,
                ls=line_style,
                zorder=2,
            )
    if len(pts) > 2:
        ax.scatter(pts[1:-1, 0], pts[1:-1, 1], s=step_size, color=arrow_color, zorder=2)
    if draw_start:
        ax.scatter(pts[0, 0], pts[0, 1], s=start_size, color=rgb0, zorder=3)


def plot_ipf_fragmentation_tracking(
    branches,
    angle_convention,
    angle_type="radians",
    direction=(0.0, 0.0, 1.0),
    crystal_symmetry="432",
    sample_symmetry="1",
    projection="stereographic",
    arrow_color="black",
    start_size=45.0,
    step_size=3.0,
    line_width=1.0,
    arrow_scale=11.0,
    label_fontsize=None,
    child_line_styles=None,
    child_colors=None,
    parent_color=None,
    parent_line_style="-",
    show_arrow=True,
    start_color=None,
    output_folder="postprocess_out",
    savefig_name="ipf_fragmentation.png",
    axis_labels=("100", "110", "111"),
    ax=None,
):
    """Plot 1->many grain fragmentation as branching IPF trajectories.

    Each branch is a parent trajectory that forks, at its split point, into
    several child trajectories -- a parent grain's orientation point splitting
    into multiple child orientation points across load steps. One IPF-colored
    start dot begins the parent line; at the parent's last (split) orientation
    each child line diverges to its own end, drawn with an arrowhead. Child lines
    are connected back to the split point so the fork is explicit.

    Args:
        branches: sequence of ``(parent_track, child_tracks)`` pairs, where
            ``parent_track`` is a ``(k, 3)`` orientation array and
            ``child_tracks`` is a sequence of ``(m_i, 3)`` orientation arrays.
            Orientations are neml2 MRP when ``angle_convention="mrp"``, else Euler.
        angle_convention: REQUIRED -- ``"mrp"`` or an Euler convention
            (``"bunge"``/``"kocks"``/``"roe"``); a wrong value silently mis-projects.
        angle_type: ``"radians"`` or ``"degrees"`` (Euler conventions only).
        direction: sample direction projected into the IPF (default z ``[0,0,1]``).
        crystal_symmetry: orbifold crystal symmetry string.
        sample_symmetry: orbifold sample symmetry string.
        projection: ``"stereographic"`` or ``"lambert"``.
        arrow_color: color of the trajectory arrows/lines. Either a single
            matplotlib color (applied to all branches), a per-branch sequence, or
            the sentinel ``"ipf_final"`` -- the parent and each child are tinted by
            the IPF color of their final-stage orientation (so a fork's color reads
            as the sub-grain's ending orientation; needs no legend).
        start_size: size of the parent start dot.
        step_size: size of the intermediate step markers.
        line_width: width of the trajectory lines and arrow shafts.
        arrow_scale: matplotlib ``mutation_scale`` for the arrowheads (head size).
        label_fontsize: font size for the triangle corner labels.
        child_line_styles: per-child matplotlib linestyles so each fragment fork
            is drawn distinctly (``"-"``/``"--"``/``"-."``/``":"``). Either a
            single style (all children), or a per-branch sequence whose ``i``-th
            item is a sequence of styles (one per child of branch ``i``); ``None``
            draws every child solid. Pair with the same fragment->style mapping in
            :func:`plot_ipf_orientation_tracking` so left/right panels reference.
        child_colors: per-child colors so a grain's sub-grains contrast (a
            categorical palette keyed by fragment). Same shape as
            ``child_line_styles`` (a per-branch sequence of per-child colors, or a
            single color); ``None`` falls back to ``arrow_color``. Overrides
            ``arrow_color`` for the children when given.
        parent_color: color for the parent (pre-split) trajectory; ``None`` uses
            ``arrow_color`` (or the ipf-final color). Pass e.g. a neutral gray when
            children are individually colored.
        parent_line_style: linestyle for the parent (pre-split) trajectory.
        show_arrow: draw an arrowhead at each child's end; ``False`` draws plain
            lines (no heads) to declutter.
        start_color: color for the parent start dot; ``None`` uses the parent's
            initial-orientation IPF color. Pass ``"black"`` for a neutral origin.
        output_folder: directory for the saved figure.
        savefig_name: output image filename.
        axis_labels: labels for the triangle corners.
        ax: existing axes to draw into; a new figure is created when ``None``.

    Returns:
        pathlib.Path: the saved figure path.
    """
    # pylint: disable=import-outside-toplevel  # heavy/optional deps (torch, neml2)
    import torch

    from .ipf_postprocess import IPFProcessor

    output_folder = Path(output_folder)
    created_fig = ax is None
    # Only touch the output folder when we actually save (i.e. no external ax).
    if created_fig:
        output_folder.mkdir(parents=True, exist_ok=True)

    ipf = IPFProcessor(
        crystal_symmetry=crystal_symmetry,
        sample_symmetry=sample_symmetry,
        projection=projection,
        save_dir=str(output_folder) if created_fig else ".",
    )
    if ax is None:
        _, ax = plt.subplots()
    ipf.ipf_color_chart(
        savefig_name=None,
        axis_labels=axis_labels,
        fill=False,
        ax=ax,
        label_fontsize=label_fontsize,
    )

    direction_t = torch.tensor(list(direction), dtype=torch.float64)
    ipf_final = isinstance(arrow_color, str) and arrow_color == "ipf_final"
    colors = None if ipf_final else _resolve_arrow_colors(arrow_color, len(branches))
    for i, (parent_track, child_tracks) in enumerate(branches):
        if child_line_styles is None:
            br_styles = ["-"] * len(child_tracks)
        elif isinstance(child_line_styles, str):
            br_styles = [child_line_styles] * len(child_tracks)
        else:
            br_styles = _resolve_line_styles(child_line_styles[i], len(child_tracks))
        br_colors = (
            _resolve_arrow_colors(child_colors[i], len(child_tracks))
            if child_colors is not None
            else None
        )
        pproj = _project_ipf_track(
            ipf, parent_track, angle_convention, angle_type, direction_t
        )
        if pproj is None:
            continue
        ppts, prgb0, prgb_final = pproj
        if parent_color is not None:
            p_color = parent_color
        elif ipf_final:
            p_color = prgb_final
        else:
            p_color = colors[i]
        dot_color = prgb0 if start_color is None else start_color
        # parent line up to the split point (no arrow -- it forks into children)
        if len(ppts) >= 2:
            ax.plot(
                ppts[:, 0],
                ppts[:, 1],
                color=p_color,
                lw=line_width,
                ls=parent_line_style,
                zorder=2,
            )
            ax.scatter(ppts[1:, 0], ppts[1:, 1], s=step_size, color=p_color, zorder=2)
        split_pt = ppts[-1]
        ax.scatter(ppts[0, 0], ppts[0, 1], s=start_size, color=dot_color, zorder=3)
        # mark the split point
        ax.scatter(split_pt[0], split_pt[1], s=step_size * 3.0, color=p_color, zorder=3)

        for j, (child, child_style) in enumerate(zip(child_tracks, br_styles)):
            cproj = _project_ipf_track(
                ipf, child, angle_convention, angle_type, direction_t
            )
            if cproj is None:
                continue
            cpts, _, crgb_final = cproj
            if br_colors is not None:
                c_color = br_colors[j]
            elif ipf_final:
                c_color = crgb_final
            else:
                c_color = colors[i]
            # connect the child back to the split point so the fork is explicit
            branch_pts = np.vstack([split_pt[None, :], cpts])
            _draw_ipf_track(
                ax,
                branch_pts,
                dot_color,
                c_color,
                start_size,
                step_size,
                draw_start=False,
                line_width=line_width,
                arrow_scale=arrow_scale,
                line_style=child_style,
                show_arrow=show_arrow,
            )

    if not created_fig:
        return ax
    fname = output_folder / savefig_name
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    return fname
