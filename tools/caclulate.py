import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.attack_methods import (
    Ibapc,
    _resolve_ibapc_noise_path,
    irba,
    nrb_door,
    pointba_i,
    pointba_o,
)
from data_utils.dataset_config import configure_dataset_args
from data_utils.ModelNetDataLoader import BDModelNetDataLoader
from model_config import get_region_data_path


PERTURBED_POINT_EPS = 1e-6


def mean_squared_nearest_neighbor_distance(source, target):
    nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nearest_neighbors.fit(target)
    distances, _ = nearest_neighbors.kneighbors(source)
    return float(np.mean(distances ** 2))


def calculate_chamfer_distance(clean_points, poisoned_points):
    """Symmetric squared Chamfer distance in the original PKL coordinates."""
    clean_points = np.asarray(clean_points, dtype=np.float32)[:, :3]
    poisoned_points = np.asarray(poisoned_points, dtype=np.float32)[:, :3]
    return (
        mean_squared_nearest_neighbor_distance(clean_points, poisoned_points)
        + mean_squared_nearest_neighbor_distance(poisoned_points, clean_points)
    )


def calculate_perturbed_point_count(clean_points, poisoned_points):
    """Number of points whose corresponding coordinates are changed."""
    clean_xyz = np.asarray(clean_points, dtype=np.float32)[:, :3]
    poisoned_xyz = np.asarray(poisoned_points, dtype=np.float32)[:, :3]
    displacement = np.linalg.norm(clean_xyz - poisoned_xyz, axis=1)
    return int(np.count_nonzero(displacement > PERTURBED_POINT_EPS))


def calculate_hausdorff_distance(clean_points, poisoned_points):
    """Symmetric Hausdorff distance in the original normalized coordinates."""
    clean_xyz = np.asarray(clean_points, dtype=np.float32)[:, :3]
    poisoned_xyz = np.asarray(poisoned_points, dtype=np.float32)[:, :3]

    poisoned_neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto")
    poisoned_neighbors.fit(poisoned_xyz)
    clean_to_poisoned, _ = poisoned_neighbors.kneighbors(clean_xyz)

    clean_neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto")
    clean_neighbors.fit(clean_xyz)
    poisoned_to_clean, _ = clean_neighbors.kneighbors(poisoned_xyz)

    return float(max(np.max(clean_to_poisoned), np.max(poisoned_to_clean)))


def set_equal_3d_axes(axes, point_clouds):
    all_xyz = np.concatenate(
        [np.asarray(points, dtype=np.float32)[:, :3] for points in point_clouds],
        axis=0,
    )
    xyz_min = np.min(all_xyz, axis=0)
    xyz_max = np.max(all_xyz, axis=0)
    center = (xyz_min + xyz_max) * 0.5
    radius = max(float(np.max(xyz_max - xyz_min)) * 0.5, 1e-6)

    for axis in axes:
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_axis_off()


def save_point_cloud_comparison(
    clean_points,
    poisoned_points,
    filename,
    attack_method,
    save_path,
):
    clean_xyz = np.asarray(clean_points, dtype=np.float32)[:, :3]
    poisoned_xyz = np.asarray(poisoned_points, dtype=np.float32)[:, :3]
    displacement = np.linalg.norm(poisoned_xyz - clean_xyz, axis=1)

    figure = plt.figure(figsize=(15, 5))
    clean_axis = figure.add_subplot(1, 3, 1, projection="3d")
    poisoned_axis = figure.add_subplot(1, 3, 2, projection="3d")
    heatmap_axis = figure.add_subplot(1, 3, 3, projection="3d")

    clean_axis.scatter(
        clean_xyz[:, 0],
        clean_xyz[:, 1],
        clean_xyz[:, 2],
        c="#2563eb",
        s=5,
        alpha=0.75,
    )
    clean_axis.set_title("Clean")

    poisoned_axis.scatter(
        poisoned_xyz[:, 0],
        poisoned_xyz[:, 1],
        poisoned_xyz[:, 2],
        c="#dc2626",
        s=5,
        alpha=0.75,
    )
    poisoned_axis.set_title(f"Poisoned ({attack_method})")

    heatmap = heatmap_axis.scatter(
        poisoned_xyz[:, 0],
        poisoned_xyz[:, 1],
        poisoned_xyz[:, 2],
        c=displacement,
        cmap="hot",
        s=7,
        alpha=0.85,
    )
    heatmap_axis.set_title("Point-wise displacement")
    colorbar = figure.colorbar(heatmap, ax=heatmap_axis, shrink=0.65, pad=0.04)
    colorbar.set_label("Point-wise displacement")

    set_equal_3d_axes(
        [clean_axis, poisoned_axis, heatmap_axis],
        [clean_xyz, poisoned_xyz],
    )
    figure.suptitle(str(filename))
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def safe_filename(value):
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    )


def resolve_test_region_path(args):
    train_region_path = args.region_data_path or str(get_region_data_path(
        args.model,
        args.num_category,
        args.dataset,
        "train",
        root=args.region_data_root,
        num_regions=16,
    ))

    if not os.path.exists(train_region_path):
        base_name = os.path.basename(train_region_path)
        split = "train" if "train" in base_name else "test"
        train_region_path = str(get_region_data_path(
            args.model,
            args.num_category,
            args.dataset,
            split,
            root=args.region_data_root,
            num_regions=16,
        ))

    return args.test_region_data_path or str(
        Path(train_region_path).parent
        / f"{args.dataset}_test_regions_with_points.pkl"
    )


def apply_comparison_attack(args, clean_points, label, sample_index=None):
    """Apply one comparison attack while preserving the original point order."""
    clean_points = np.asarray(clean_points, dtype=np.float32)
    attack_points = clean_points[None, :, :3].copy()
    attack_labels = np.asarray([[label]], dtype=np.int64)
    selected_indices = frozenset([0])

    attack_method = args.attack_method.lower()
    had_previous_index = hasattr(args, "ibapc_sample_index")
    previous_index = getattr(args, "ibapc_sample_index", None)
    if sample_index is not None:
        args.ibapc_sample_index = int(sample_index)
    try:
        if attack_method == "pointba_i":
            pointba_i(
                attack_points,
                attack_labels,
                selected_indices,
                args.target_label,
            )
        elif attack_method == "pointba_o":
            pointba_o(
                attack_points,
                attack_labels,
                selected_indices,
                args.target_label,
            )
        elif attack_method == "nrb_door":
            nrb_door(
                attack_points,
                attack_labels,
                selected_indices,
                args.target_label,
            )
        elif attack_method == "irba":
            irba(args, attack_points, attack_labels, selected_indices)
        elif attack_method == "ibapc":
            Ibapc(args, attack_points, attack_labels, selected_indices)
        else:
            raise ValueError(f"Unsupported comparison attack: {args.attack_method}")
    finally:
        if sample_index is not None:
            if had_previous_index:
                args.ibapc_sample_index = previous_index
            elif hasattr(args, "ibapc_sample_index"):
                delattr(args, "ibapc_sample_index")

    poisoned_points = clean_points.copy()
    poisoned_points[:, :3] = attack_points[0]
    return poisoned_points


class RegionBATriggerApplicator(BDModelNetDataLoader):
    """Reuse RegionBA selection and perturbation without dataset side effects."""
    def __init__(self, args):
        self.args = args
        self.num_category = args.num_category
        self.attack_region_idx = getattr(args, "attack_region_idx", None)


def build_clean_poisoned_pairs(args):
    with open(args.region_data_path, "rb") as file:
        region_data = pickle.load(file)

    regionba_trigger = (
        RegionBATriggerApplicator(args)
        if args.attack_method == "regionba"
        else None
    )
    region_items = list(region_data.items())
    if args.attack_method == "ibapc":
        region_items = tqdm(
            list(enumerate(region_items)),
            desc="Applying IBAPC trigger",
        )
    else:
        region_items = enumerate(region_items)

    pairs = []
    for sample_index, (filename, sample_data) in region_items:
        label = int(sample_data["label"])
        if label == args.target_label:
            continue
        clean_points = np.asarray(sample_data["points"], dtype=np.float32)
        if regionba_trigger is not None:
            _, attack_region = regionba_trigger.select_attack_regions(
                clean_points,
                sample_data["regions"],
                region_scores=sample_data.get("scores", None),
            )
            poisoned_points = regionba_trigger.add_region_perturbation(
                clean_points.copy(),
                attack_region,
            )
        else:
            poisoned_points = apply_comparison_attack(
                args,
                clean_points,
                label,
                sample_index=sample_index,
            )
        pairs.append((filename, clean_points, poisoned_points))
    return pairs


def calculate_distances(args):
    configure_dataset_args(args)
    args.attack_method = args.attack_method.lower()
    args.region_data_path = resolve_test_region_path(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    clean_poisoned_pairs = build_clean_poisoned_pairs(args)
    visualization_count = min(
        max(int(args.visualize_samples), 0),
        len(clean_poisoned_pairs),
    )
    if visualization_count > 0:
        visualization_indices = set(
            np.linspace(
                0,
                len(clean_poisoned_pairs) - 1,
                num=visualization_count,
                dtype=np.int64,
            ).tolist()
        )
        visualization_dir = (
            Path(args.visualization_dir)
            / args.dataset
            / args.model
            / args.attack_method
        )
        visualization_dir.mkdir(parents=True, exist_ok=True)
    else:
        visualization_indices = set()
        visualization_dir = None

    chamfer_distances = []
    perturbed_point_counts = []
    saved_visualizations = 0
    for pair_index, (filename, clean_points, poisoned_points) in enumerate(
        tqdm(
            clean_poisoned_pairs,
            desc="Calculating CD and NP",
        )
    ):
        chamfer_distances.append(
            calculate_chamfer_distance(clean_points, poisoned_points)
        )
        perturbed_point_counts.append(
            calculate_perturbed_point_count(clean_points, poisoned_points)
        )
        if pair_index in visualization_indices:
            saved_visualizations += 1
            image_name = (
                f"{saved_visualizations:02d}_"
                f"{safe_filename(filename)}.png"
            )
            save_point_cloud_comparison(
                clean_points,
                poisoned_points,
                filename=filename,
                attack_method=args.attack_method,
                save_path=visualization_dir / image_name,
            )

    if not chamfer_distances:
        raise ValueError("No non-target test samples were available.")

    mean_cd = float(np.mean(chamfer_distances))
    mean_np = int(round(float(np.mean(perturbed_point_counts))))
    min_np = int(np.min(perturbed_point_counts))
    max_np = int(np.max(perturbed_point_counts))
    total_np = int(np.sum(perturbed_point_counts))
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Attack method: {args.attack_method}")
    print(f"Test region pkl: {args.region_data_path}")
    print(f"Seed: {args.seed}")
    print(f"Target label: {args.target_label}")
    if args.attack_method == "regionba":
        print(f"Attack region mode: {args.attack_region_mode}")
        print(f"Grid density: {args.grid_density}")
    elif args.attack_method == "irba":
        print(f"IRBA anchors: {args.num_anchor}")
        print(f"IRBA rotation: {args.R_alpha}")
        print(f"IRBA scale: {args.S_size}")
    elif args.attack_method == "ibapc":
        print(f"IBAPC noise path: {_resolve_ibapc_noise_path(args)}")
        print(f"IBAPC kNN: {args.ibapc_knn}")
        print(f"IBAPC eigen cache root: {args.ibapc_eigen_cache_root or '/root/shared-nvme/eigen_cache if available'}")
        sample_tag = "fps" if getattr(args, "use_uniform_sample", False) else "normal"
        cache_root = Path(args.ibapc_eigen_cache_root or "/root/shared-nvme/eigen_cache")
        print(
            "IBAPC test eigen cache:",
            cache_root
            / args.dataset
            / f"test_n{args.num_point}_{sample_tag}_knn{args.ibapc_knn}",
        )
    print(f"Poisoned test samples: {len(clean_poisoned_pairs)}")
    print(f"Chamfer Distance: {mean_cd:.8f}")
    print(f"Chamfer Distance (x1000): {mean_cd * 1000.0:.4f}")
    print(f"Perturbed Points (avg): {mean_np}")
    print(f"Perturbed Points (min/max): {min_np}/{max_np}")
    print(f"Perturbed Points (total): {total_np}")
    if visualization_dir is not None:
        print(f"Visualized samples: {saved_visualizations}")
        print(f"Visualization directory: {visualization_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        "Calculate CD and NP for point-cloud attacks"
    )
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model", type=str, default="dgcnn")
    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument(
        "--num_category", default=None, type=int, choices=[10, 16, 40]
    )
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--use_uniform_sample", action="store_true", default=True)
    parser.add_argument("--process_data", action="store_true", default=False)

    parser.add_argument("--poisoned_rate", type=float, default=0.05)
    parser.add_argument("--target_label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument(
        "--attack_method",
        type=str,
        default="regionba",
        choices=["regionba", "pointba_i", "pointba_o", "nrb_door", "irba", "ibapc"],
        help="attack used to construct poisoned test point clouds",
    )
    parser.add_argument("--grid_density", type=float, default=0.4)
    parser.add_argument("--num_anchor", type=int, default=16)
    parser.add_argument("--R_alpha", type=float, default=5.0)
    parser.add_argument("--S_size", type=float, default=5.0)
    parser.add_argument(
        "--ibapc_noise_path",
        type=str,
        default=None,
        help="IBAPC GFT_noise.npy path; default uses data_utils/ibapc/noises/{dataset}/{model}/target*_seed*/GFT_noise.npy",
    )
    parser.add_argument("--ibapc_knn", type=int, default=10)
    parser.add_argument(
        "--ibapc_eigen_cache_root",
        type=str,
        default=None,
        help="cache root for IBAPC test eigenvectors; default prefers /root/shared-nvme/eigen_cache",
    )
    parser.add_argument(
        "--attack_region_mode",
        type=str,
        default="top4",
        choices=[
            "top1", "top2", "top4", "top6", "top8", "top10", "top12", "top14", "top16",
            "bottom1", "bottom2", "bottom4", "random2_connected",
        ],
    )
    parser.add_argument("--region_data_path", type=str, default=None)
    parser.add_argument("--test_region_data_path", type=str, default=None)
    parser.add_argument("--region_data_root", type=str, default="data")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=10,
        help="number of evenly spaced poisoned samples to visualize; 0 disables",
    )
    parser.add_argument(
        "--visualization_dir",
        type=str,
        default="visualization/attack_method_comparisons",
    )
    return parser.parse_args()


if __name__ == "__main__":
    calculate_distances(parse_args())

