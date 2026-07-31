import numpy as np
from sklearn.neighbors import NearestNeighbors


# EDGE_STRETCH_ALPHA = 0.1


def add_region_perturbation(
    points,
    attack_region,
    grid_density=0.45,
):
    perturbed = points.copy()
    attack_region = np.asarray(attack_region, dtype=np.int32).reshape(-1)

    whole_xyz = perturbed[:, :3].astype(np.float64, copy=True)
    region_xyz = whole_xyz[attack_region]
    density = float(grid_density)
    if density <= 0:
        raise ValueError("grid_density must be greater than 0")

    whole_min = np.min(whole_xyz, axis=0)
    whole_max = np.max(whole_xyz, axis=0)
    whole_center = (whole_min + whole_max) * 0.5
    cube_side = float(np.max(whole_max - whole_min))

    nn_global = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_global.fit(whole_xyz)
    d_global, _ = nn_global.kneighbors(whole_xyz)
    base_spacing = float(np.median(d_global[:, 1]))

    grid_spacing = base_spacing / density
    cube_pad = grid_spacing
    cube_min = whole_center - cube_side * 0.5 - cube_pad
    region_center = np.mean(region_xyz, axis=0)
    used_nodes = set()

    def snap_midpoint_to_grid(midpoint):
        grid_position = (midpoint - cube_min) / grid_spacing
        nearest_index = np.rint(grid_position).astype(np.int64)
        fractional_offset = grid_position - nearest_index
        search_radius = 0

        while True:
            axis_offsets = np.arange(
                -search_radius,
                search_radius + 1,
                dtype=np.int64,
            )
            offsets = np.stack(
                np.meshgrid(
                    axis_offsets,
                    axis_offsets,
                    axis_offsets,
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 3)
            candidate_indices = nearest_index[None, :] + offsets
            available_mask = np.array(
                [
                    tuple(int(v) for v in index) not in used_nodes
                    for index in candidate_indices
                ],
                dtype=bool,
            )

            if np.any(available_mask):
                available_indices = candidate_indices[available_mask]
                available_points = (
                    cube_min[None, :]
                    + available_indices.astype(np.float64) * grid_spacing
                )
                distances = np.linalg.norm(
                    available_points - midpoint[None, :],
                    axis=1,
                )
                best_position = int(np.argmin(distances))
                best_distance = float(distances[best_position])
                outside_lower_bound = (
                    search_radius
                    + 1
                    - float(np.max(np.abs(fractional_offset)))
                ) * grid_spacing
                if best_distance <= outside_lower_bound + 1e-12:
                    best_index = available_indices[best_position]
                    used_nodes.add(tuple(int(v) for v in best_index))
                    return available_points[best_position]

            search_radius += 1

    local_order = np.argsort(
        np.linalg.norm(region_xyz - region_center[None, :], axis=1)
    )
    local_nn = NearestNeighbors(
        n_neighbors=min(region_xyz.shape[0], 2),
        algorithm="auto",
    )
    local_nn.fit(region_xyz)
    _, local_neighbors = local_nn.kneighbors(region_xyz)

    remaining = set(int(i) for i in range(region_xyz.shape[0]))
    pairs = []
    for local_i in local_order:
        local_i = int(local_i)
        if local_i not in remaining:
            continue

        remaining.remove(local_i)
        mate = None
        for local_j in local_neighbors[local_i, 1:]:
            local_j = int(local_j)
            if local_j in remaining:
                mate = local_j
                break

        if mate is None and len(remaining) > 0:
            rest = np.array(sorted(remaining), dtype=np.int32)
            rest_dist = np.linalg.norm(
                region_xyz[rest] - region_xyz[local_i][None, :],
                axis=1,
            )
            mate = int(rest[int(np.argmin(rest_dist))])

        if mate is None:
            break

        remaining.remove(mate)
        pairs.append((int(attack_region[local_i]), int(attack_region[mate])))

    moved_xyz = whole_xyz.copy()
    for idx0, idx1 in pairs:
        p0 = whole_xyz[idx0]
        p1 = whole_xyz[idx1]
        midpoint = (p0 + p1) * 0.5
        snapped_mid = snap_midpoint_to_grid(midpoint)

        segment = p1 - p0
        moved_xyz[idx0] = snapped_mid - segment * 0.5
        moved_xyz[idx1] = snapped_mid + segment * 0.5
        # stretched_segment = segment * (1.0 + EDGE_STRETCH_ALPHA)
        # moved_xyz[idx0] = snapped_mid - stretched_segment * 0.5
        # moved_xyz[idx1] = snapped_mid + stretched_segment * 0.5

    perturbed[:, :3] = moved_xyz.astype(perturbed.dtype, copy=False)
    return perturbed
