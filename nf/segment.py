import torch

import tqdm

from nf.image import get_neighbor_indices, connectivity_options
from nf import metrics


def flood(
    angles,
    phase,
    misorientation_tol,
    connectivity=26,
    batch_norm=10000000,
    grain_threshold=100,
    stop_count=100,
    **kwargs
):
    """Flood fill algorithm to segment connected components in 3D data

    Args:
        angles (nx,ny,nz,3) array: Euler angles of input, interpreted according to kwargs of metrics.misorientation
        phase (nx,ny,nz) array: phase identifiers for each voxel
        misorientation_tol (float): misorientation tolerance for connectivity

    Keyword Args:
        connectivity (int): 6 or 26 connectivity for neighbors
        batch_norm (int): batch size for calculating distance
        grain_threshold (int): minimum size of a segment to be kept
        stop_count (int): stop after finding this many small segments
        **kwargs: additional arguments for metrics.misorientation(), such as angle_convention

    Returns:
        segmented (nx,ny,nz) array: segmented phases
    """
    # Get the neighbor indices
    offsets = connectivity_options[connectivity]
    Xk, Yk, Zk, valid = get_neighbor_indices(offsets, angles.shape[:3])

    # Precompute distances
    distances = metrics.batched_norm(
        angles.unsqueeze(0).expand(Xk.shape[0], -1, -1, -1, -1).reshape(-1, 3),
        angles[Xk, Yk, Zk].reshape(-1, 3),
        metrics.misorientation,
        chunk_size=batch_norm,
        **kwargs
    ).reshape(Xk.shape)

    # Mark invalid distances as infinity so we don't flood them
    distances.masked_fill(~valid, float("inf"))

    # Initial setup:
    # Phase == 1 is valid material, Phase == 0 is void, we only want to segment phase == 1
    phase[phase > 0] = -1

    # Flood fill segmentation
    current_segment = 1
    with tqdm.tqdm(total=torch.sum(phase < 0).item(), desc="Segmenting") as pbar:
        while True:
            # Find unsegmented voxels (phase < 0)
            unsegmented = torch.where(phase < 0)
            if unsegmented[0].shape[0] == 0:
                break

            # Pick a random unsegmented voxel as seed
            choice = torch.randint(0, unsegmented[0].shape[0], (1,)).item()
            seed = torch.stack(
                [unsegmented[0][choice], unsegmented[1][choice], unsegmented[2][choice]]
            )

            # Initialize the front with the seed voxel
            front = seed.unsqueeze(0)

            # Flood fill from this seed
            while front.shape[0] > 0:
                # Get current voxel coordinates
                x, y, z = front[:, 0], front[:, 1], front[:, 2]

                # Mark current voxels as part of this segment
                phase[x, y, z] = current_segment

                # Get all neighbors of current front voxels
                neighbor_x = Xk[:, x, y, z].reshape(-1)
                neighbor_y = Yk[:, x, y, z].reshape(-1)
                neighbor_z = Zk[:, x, y, z].reshape(-1)

                # Check which neighbors are unsegmented (phase < 0)
                neighbor_unsegmented = phase[neighbor_x, neighbor_y, neighbor_z] < 0

                # Get distances to neighbors
                neighbor_distance = distances[:, x, y, z].reshape(-1)

                # Keep only unsegmented neighbors within misorientation tolerance
                mask = neighbor_unsegmented & (neighbor_distance < misorientation_tol)
                neighbor_x = neighbor_x[mask]
                neighbor_y = neighbor_y[mask]
                neighbor_z = neighbor_z[mask]

                # Remove duplicate neighbors
                if neighbor_x.shape[0] > 0:
                    front = torch.unique(
                        torch.stack([neighbor_x, neighbor_y, neighbor_z], dim=1), dim=0
                    )
                else:
                    front = torch.empty((0, 3), dtype=torch.long)

            size_current = torch.sum(phase == current_segment).item()
            
            if size_current < grain_threshold:
                # Remove small segments
                phase[phase == current_segment] = -1
                stop_count -= 1
                if stop_count == 0:
                    break
            else:
                # Update progress bar
                pbar.update(size_current)
                current_segment += 1

    return phase


def infill_nearest_neighbor(grid, connectivity=6):
    """Infill unmarked voxels (phase == -1) with nearest neighbor phases

    Args:
        grid (nx,ny,nz,7) tensor: fixed grid format with grid[...,0] being phase index
        connectivity (int): 6 or 26 connectivity for neighbors

    Returns:
        grid (nx,ny,nz,7) tensor: grid with all phase == -1 voxels filled
    """
    grid = grid.clone()
    phase = grid[..., 0]

    # Get neighbor indices
    offsets = connectivity_options[connectivity]
    Xk, Yk, Zk, valid = get_neighbor_indices(offsets, phase.shape)

    with tqdm.tqdm(total=torch.sum(phase == -1).item(), desc="Infilling") as pbar:
        while True:
            # Find voxels that need infilling
            unfilled_mask = phase == -1
            unfilled_count = torch.sum(unfilled_mask).item()

            if unfilled_count == 0:
                break

            # Get neighbors of unfilled voxels
            unfilled_x, unfilled_y, unfilled_z = torch.where(unfilled_mask)

            # For each unfilled voxel, check its neighbors
            neighbor_x = Xk[
                :, unfilled_x, unfilled_y, unfilled_z
            ]  # (n_offsets, n_unfilled)
            neighbor_y = Yk[:, unfilled_x, unfilled_y, unfilled_z]
            neighbor_z = Zk[:, unfilled_x, unfilled_y, unfilled_z]
            neighbor_valid = valid[:, unfilled_x, unfilled_y, unfilled_z]

            # Get phase values of neighbors
            neighbor_phases = phase[
                neighbor_x, neighbor_y, neighbor_z
            ]  # (n_offsets, n_unfilled)

            # Mark invalid neighbors as -1
            neighbor_phases = neighbor_phases.where(
                neighbor_valid, torch.tensor(-1, dtype=neighbor_phases.dtype)
            )

            # Find which unfilled voxels have at least one filled neighbor (phase >= 1)
            has_filled_neighbor = torch.any(neighbor_phases >= 1, dim=0)

            # For voxels with filled neighbors, assign the first filled neighbor's phase
            voxels_to_fill = torch.where(has_filled_neighbor)[0]

            if voxels_to_fill.shape[0] == 0:
                # No progress possible, break to avoid infinite loop
                break

            for idx in voxels_to_fill:
                x, y, z = unfilled_x[idx], unfilled_y[idx], unfilled_z[idx]
                # Find first valid filled neighbor
                neighbor_mask = neighbor_phases[:, idx] >= 1
                if neighbor_mask.any():
                    first_filled_idx = torch.where(neighbor_mask)[0][0]
                    nx, ny, nz = (
                        neighbor_x[first_filled_idx, idx],
                        neighbor_y[first_filled_idx, idx],
                        neighbor_z[first_filled_idx, idx],
                    )
                    # Copy all attributes from the neighbor
                    grid[x, y, z, 0:4] = grid[nx, ny, nz, 0:4]

            pbar.update(voxels_to_fill.shape[0])

    return grid


def remove_small_segments(grid, min_size, connectivity=6):
    """Remove small segments by merging them into nearest larger segments

    Args:
        grid (nx,ny,nz,7) tensor: fixed grid format with grid[...,0] being phase index
        min_size (int): minimum segment size in voxels; segments smaller than this are removed
        connectivity (int): 6 or 26 connectivity for neighbors

    Returns:
        grid (nx,ny,nz,7) tensor: grid with small segments merged into neighbors
    """
    grid = grid.clone()
    phase = grid[..., 0]

    # Get neighbor indices
    offsets = connectivity_options[connectivity]
    Xk, Yk, Zk, valid = get_neighbor_indices(offsets, phase.shape)

    # Find all unique segments (excluding void phase 0 and unfilled phase -1)
    unique_segments = torch.unique(phase[phase >= 1])

    # Identify small segments
    small_segments = []
    for seg in unique_segments:
        seg_size = torch.sum(phase == seg).item()
        if seg_size < min_size and seg != 0:
            small_segments.append(seg.item())

    if len(small_segments) == 0:
        return grid

    for small_seg in tqdm.tqdm(small_segments, desc="Removing small segments"):
        # Find all voxels in this small segment
        seg_x, seg_y, seg_z = torch.where(phase == small_seg)

        if seg_x.shape[0] == 0:
            continue

        # Get all neighbors of this segment
        neighbor_x = Xk[:, seg_x, seg_y, seg_z]  # (n_offsets, n_voxels)
        neighbor_y = Yk[:, seg_x, seg_y, seg_z]
        neighbor_z = Zk[:, seg_x, seg_y, seg_z]
        neighbor_valid = valid[:, seg_x, seg_y, seg_z]

        # Get phase values of neighbors
        neighbor_phases = phase[neighbor_x, neighbor_y, neighbor_z]

        # Filter to only valid neighbors that are larger segments (not this segment, not void, not unfilled)
        neighbor_is_valid_segment = (
            neighbor_valid & (neighbor_phases >= 1) & (neighbor_phases != small_seg)
        )

        # Find unique neighboring segments and count contacts
        if neighbor_is_valid_segment.any():
            valid_neighbor_phases = neighbor_phases[neighbor_is_valid_segment]
            unique_neighbors, counts = torch.unique(
                valid_neighbor_phases, return_counts=True
            )

            # Choose the neighbor with most contact points
            if unique_neighbors.shape[0] > 0:
                most_common_neighbor = unique_neighbors[torch.argmax(counts)]

                # Find a representative voxel from the target segment
                target_x, target_y, target_z = torch.where(
                    phase == most_common_neighbor
                )
                if target_x.shape[0] > 0:
                    # Copy attributes from first voxel of target segment to all voxels of small segment
                    target_values = grid[target_x[0], target_y[0], target_z[0]].clone()
                    grid[seg_x, seg_y, seg_z, 0:4] = target_values[0:4]

    return grid
