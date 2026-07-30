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

"""Flood-fill grain segmentation and cleanup for NF voxel grids."""

from __future__ import annotations

import torch

import tqdm

from .image import get_neighbor_indices, connectivity_options
from . import metrics


def flood(
    angles,
    phase,
    misorientation_tol,
    connectivity=26,
    batch_norm=10000000,
    grain_threshold=100,
    stop_count=100,
    **kwargs,
):
    """Flood-fill segmentation of connected components in a 3D voxel grid.

    Args:
        angles (nx,ny,nz,3): Euler angles, interpreted per metrics.misorientation kwargs
        phase (nx,ny,nz): phase identifiers per voxel (>0 material, 0 void)
        misorientation_tol (float): misorientation tolerance for connectivity
        connectivity (int): 6 or 26
        batch_norm (int): batch size for distance calculation
        grain_threshold (int): minimum segment size to keep
        stop_count (int): stop after finding this many small segments
        **kwargs: extra arguments for metrics.misorientation()

    Returns:
        (nx,ny,nz) array of segmented phases
    """
    offsets = connectivity_options[connectivity]
    Xk, Yk, Zk, valid = get_neighbor_indices(offsets, angles.shape[:3])

    distances = metrics.batched_norm(
        angles.unsqueeze(0).expand(Xk.shape[0], -1, -1, -1, -1).reshape(-1, 3),
        angles[Xk, Yk, Zk].reshape(-1, 3),
        metrics.misorientation,
        chunk_size=batch_norm,
        **kwargs,
    ).reshape(Xk.shape)

    # Mark invalid distances as infinity so we don't flood them
    distances.masked_fill(~valid, float("inf"))

    # Phase > 0 is material to segment, 0 is void; mark material unsegmented (-1)
    phase[phase > 0] = -1

    current_segment = 1
    with tqdm.tqdm(total=torch.sum(phase < 0).item(), desc="Segmenting") as pbar:
        while True:
            unsegmented = torch.where(phase < 0)
            if unsegmented[0].shape[0] == 0:
                break

            # Pick a random unsegmented voxel as seed
            choice = torch.randint(0, unsegmented[0].shape[0], (1,)).item()
            seed = torch.stack(
                [unsegmented[0][choice], unsegmented[1][choice], unsegmented[2][choice]]
            )

            front = seed.unsqueeze(0)

            while front.shape[0] > 0:
                x, y, z = front[:, 0], front[:, 1], front[:, 2]

                phase[x, y, z] = current_segment

                neighbor_x = Xk[:, x, y, z].reshape(-1)
                neighbor_y = Yk[:, x, y, z].reshape(-1)
                neighbor_z = Zk[:, x, y, z].reshape(-1)

                neighbor_unsegmented = phase[neighbor_x, neighbor_y, neighbor_z] < 0

                neighbor_distance = distances[:, x, y, z].reshape(-1)

                # Keep unsegmented neighbors within misorientation tolerance
                mask = neighbor_unsegmented & (neighbor_distance < misorientation_tol)
                neighbor_x = neighbor_x[mask]
                neighbor_y = neighbor_y[mask]
                neighbor_z = neighbor_z[mask]

                if neighbor_x.shape[0] > 0:
                    front = torch.unique(
                        torch.stack([neighbor_x, neighbor_y, neighbor_z], dim=1), dim=0
                    )
                else:
                    front = torch.empty((0, 3), dtype=torch.long)

            size_current = torch.sum(phase == current_segment).item()

            if size_current < grain_threshold:
                # Discard segments below the threshold
                phase[phase == current_segment] = -1
                stop_count -= 1
                if stop_count == 0:
                    break
            else:
                pbar.update(size_current)
                current_segment += 1

    return phase


def infill_nearest_neighbor(grid, connectivity=6):
    """Infill unmarked voxels (phase == -1) from nearest filled neighbors.

    Args:
        grid (nx,ny,nz,7) tensor: fixed grid, grid[...,0] is the phase index
        connectivity (int): 6 or 26

    Returns:
        (nx,ny,nz,7) tensor with all phase == -1 voxels filled
    """
    grid = grid.clone()
    phase = grid[..., 0]

    offsets = connectivity_options[connectivity]
    Xk, Yk, Zk, valid = get_neighbor_indices(offsets, phase.shape)

    with tqdm.tqdm(total=torch.sum(phase == -1).item(), desc="Infilling") as pbar:
        while True:
            unfilled_mask = phase == -1
            unfilled_count = torch.sum(unfilled_mask).item()

            if unfilled_count == 0:
                break

            unfilled_x, unfilled_y, unfilled_z = torch.where(unfilled_mask)

            neighbor_x = Xk[:, unfilled_x, unfilled_y, unfilled_z]
            neighbor_y = Yk[:, unfilled_x, unfilled_y, unfilled_z]
            neighbor_z = Zk[:, unfilled_x, unfilled_y, unfilled_z]
            neighbor_valid = valid[:, unfilled_x, unfilled_y, unfilled_z]

            neighbor_phases = phase[neighbor_x, neighbor_y, neighbor_z]

            # Treat out-of-bounds neighbors as unfilled
            neighbor_phases = neighbor_phases.where(
                neighbor_valid, torch.tensor(-1, dtype=neighbor_phases.dtype)
            )

            has_filled_neighbor = torch.any(neighbor_phases >= 1, dim=0)

            voxels_to_fill = torch.where(has_filled_neighbor)[0]

            if voxels_to_fill.shape[0] == 0:
                # No progress possible; avoid an infinite loop
                break

            for idx in voxels_to_fill:
                x, y, z = unfilled_x[idx], unfilled_y[idx], unfilled_z[idx]
                neighbor_mask = neighbor_phases[:, idx] >= 1
                if neighbor_mask.any():
                    first_filled_idx = torch.where(neighbor_mask)[0][0]
                    nx, ny, nz = (
                        neighbor_x[first_filled_idx, idx],
                        neighbor_y[first_filled_idx, idx],
                        neighbor_z[first_filled_idx, idx],
                    )
                    grid[x, y, z, 0:4] = grid[nx, ny, nz, 0:4]

            pbar.update(voxels_to_fill.shape[0])

    return grid


def remove_small_segments(grid, min_size, connectivity=6):
    """Merge segments smaller than ``min_size`` into their largest-contact neighbor.

    Args:
        grid (nx,ny,nz,7) tensor: fixed grid, grid[...,0] is the phase index
        min_size (int): minimum segment size in voxels
        connectivity (int): 6 or 26

    Returns:
        (nx,ny,nz,7) tensor with small segments merged into neighbors
    """
    grid = grid.clone()
    phase = grid[..., 0]

    offsets = connectivity_options[connectivity]
    Xk, Yk, Zk, valid = get_neighbor_indices(offsets, phase.shape)

    unique_segments = torch.unique(phase[phase >= 1])

    small_segments = []
    for seg in unique_segments:
        seg_size = torch.sum(phase == seg).item()
        if seg_size < min_size and seg != 0:
            small_segments.append(seg.item())

    if len(small_segments) == 0:
        return grid

    for small_seg in tqdm.tqdm(small_segments, desc="Removing small segments"):
        seg_x, seg_y, seg_z = torch.where(phase == small_seg)

        if seg_x.shape[0] == 0:
            continue

        neighbor_x = Xk[:, seg_x, seg_y, seg_z]
        neighbor_y = Yk[:, seg_x, seg_y, seg_z]
        neighbor_z = Zk[:, seg_x, seg_y, seg_z]
        neighbor_valid = valid[:, seg_x, seg_y, seg_z]

        neighbor_phases = phase[neighbor_x, neighbor_y, neighbor_z]

        # Keep valid neighbors belonging to a different real segment
        neighbor_is_valid_segment = (
            neighbor_valid & (neighbor_phases >= 1) & (neighbor_phases != small_seg)
        )

        if neighbor_is_valid_segment.any():
            valid_neighbor_phases = neighbor_phases[neighbor_is_valid_segment]
            unique_neighbors, counts = torch.unique(
                valid_neighbor_phases, return_counts=True
            )

            if unique_neighbors.shape[0] > 0:
                # Merge into the neighbor with the most contact points
                most_common_neighbor = unique_neighbors[torch.argmax(counts)]

                target_x, target_y, target_z = torch.where(
                    phase == most_common_neighbor
                )
                if target_x.shape[0] > 0:
                    target_values = grid[target_x[0], target_y[0], target_z[0]].clone()
                    grid[seg_x, seg_y, seg_z, 0:4] = target_values[0:4]
                else:
                    grid[seg_x, seg_y, seg_z, 0] = -1
            else:
                grid[seg_x, seg_y, seg_z, 0] = -1
        else:
            grid[seg_x, seg_y, seg_z, 0] = -1

    return grid
