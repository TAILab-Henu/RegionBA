import argparse
import csv
import math
import os
import pickle
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
from tqdm import tqdm

sys.path.append(str(BASE_DIR / "models"))

from data_utils.attack_methods import pointba_i
from data_utils.dataset_config import (
    configure_dataset_args,
    create_backdoor_dataset,
    create_clean_dataset,
)
from defense import build_model
from model_config import get_region_data_path


ATTACK_METHODS = ["regionba", "pointba_i"]
ATTACK_REGION_MODES = [
    "top1",
    "top2",
    "top4",
    "top6",
    "top8",
    "top10",
    "top12",
    "top14",
    "top16",
    "bottom1",
    "bottom2",
    "bottom4",
    "random2_connected",
]


def parse_args():
    parser = argparse.ArgumentParser("STRIP defense test for point-cloud backdoors")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--model", default="dgcnn")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_category", default=None, type=int, choices=[10, 16, 40])
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--use_uniform_sample", action="store_true", default=True)

    parser.add_argument("--target_label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument(
        "--attack_method",
        type=str,
        default="regionba",
        choices=ATTACK_METHODS,
    )
    parser.add_argument("--grid_density", type=float, default=0.4)
    parser.add_argument(
        "--attack_region_mode",
        type=str,
        default="top4",
        choices=ATTACK_REGION_MODES,
    )
    parser.add_argument("--attack_region_idx", type=int, default=None)
    parser.add_argument("--region_data_path", type=str, default=None)
    parser.add_argument("--region_data_root", type=str, default="data")

    parser.add_argument(
        "--num_eval",
        type=int,
        default=2000,
        help="number of benign/triggered non-target samples; STRIP uses 2000",
    )
    parser.add_argument(
        "--num_mix",
        type=int,
        default=100,
        help="number of benign fusion samples per input; STRIP uses 100",
    )
    parser.add_argument(
        "--frr",
        type=float,
        default=0.01,
        help="benign false rejection rate used to set the entropy threshold",
    )
    parser.add_argument(
        "--overlay_split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="clean benign split used as STRIP overlays",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="number of histogram bins",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="concat",
        choices=["concat_sample", "concat"],
        help=(
            "STRIP point-cloud fusion. concat keeps all points from the detected "
            "and overlay clouds, matching the point-cloud STRIP/PointCRT setting; "
            "concat_sample samples the fused cloud back to the original point count."
        ),
    )
    parser.add_argument(
        "--entropy_mode",
        type=str,
        default="softmax_norm",
        choices=["softmax_norm", "sigmoid", "softmax"],
        help=(
            "entropy calculation. softmax_norm reports softmax entropy normalized "
            "to [0, 100], matching the normalized-entropy axis used in STRIP plots; "
            "sigmoid and softmax are kept only for ablation."
        ),
    )
    parser.add_argument(
        "--ylim",
        type=float,
        default=0.0,
        help="fixed y-axis upper bound for the entropy histogram; use <=0 for auto",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_numpy_points(points):
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    return np.asarray(points, dtype=np.float32)


def collect_clean_samples(dataset, target_label=None, limit=None, exclude_target=False):
    points_list = []
    labels_list = []
    filenames = []
    all_filenames = getattr(dataset, "filenames", None)

    for index in range(len(dataset)):
        points, label = dataset[index]
        label = int(label)
        if exclude_target and target_label is not None and label == int(target_label):
            continue
        points_list.append(to_numpy_points(points).copy())
        labels_list.append(label)
        if all_filenames is not None and index < len(all_filenames):
            filenames.append(str(all_filenames[index]))
        else:
            filenames.append(str(index))
        if limit is not None and len(points_list) >= int(limit):
            break

    return points_list, labels_list, filenames


def collect_regionba_samples(args, limit):
    args.disable_perturbation_visualization = True
    args.max_detail_logs = 0
    dataset = create_backdoor_dataset(args, split="test")

    points_list = []
    labels_list = []
    filenames = getattr(dataset, "filenames", None)
    selected_filenames = []

    for index in range(len(dataset)):
        points, label = dataset[index]
        points_list.append(to_numpy_points(points).copy())
        labels_list.append(int(label))
        if filenames is not None and index < len(filenames):
            selected_filenames.append(str(filenames[index]))
        else:
            selected_filenames.append(str(index))
        if limit is not None and len(points_list) >= int(limit):
            break

    return points_list, labels_list, selected_filenames


def collect_region_pkl_clean_samples(args, limit):
    with open(args.region_data_path, "rb") as file:
        region_data = pickle.load(file)

    points_list = []
    labels_list = []
    filenames = []
    for filename, sample_data in region_data.items():
        label = int(sample_data["label"])
        if label == int(args.target_label):
            continue
        points_list.append(np.asarray(sample_data["points"], dtype=np.float32).copy())
        labels_list.append(label)
        filenames.append(str(filename))
        if limit is not None and len(points_list) >= int(limit):
            break

    return points_list, labels_list, filenames


def apply_pointba_i(points, label, target_label):
    poisoned = np.asarray(points, dtype=np.float32).copy()
    attack_points = poisoned[None, :, :3].copy()
    attack_labels = np.asarray([[label]], dtype=np.int64)
    pointba_i(attack_points, attack_labels, frozenset([0]), int(target_label))
    poisoned[:, :3] = attack_points[0]
    return poisoned, int(attack_labels[0, 0])


def collect_pointba_i_samples(clean_points, clean_labels, target_label):
    poisoned_points = []
    poisoned_labels = []
    for points, label in zip(clean_points, clean_labels):
        poisoned, poisoned_label = apply_pointba_i(points, label, target_label)
        poisoned_points.append(poisoned)
        poisoned_labels.append(poisoned_label)
    return poisoned_points, poisoned_labels


def strip_fuse(background, overlay, rng, fusion_mode):
    """Point-cloud analogue of STRIP sample fusion.

    For images, STRIP superimposes two same-size arrays and therefore keeps the
    whole detected sample, including a possible trigger. Direct coordinate-wise
    addition is not meaningful for unordered point clouds and can push the
    object far outside the training distribution. We instead fuse two point
    clouds by concatenating their points. The default mode samples the fused
    cloud back to the original point count, so STRIP applies a strong
    perturbation without changing the model input size.
    """
    background = np.asarray(background, dtype=np.float32)
    overlay = np.asarray(overlay, dtype=np.float32)
    if overlay.shape[1] != background.shape[1]:
        raise ValueError(
            f"STRIP fusion expects equal point channels: "
            f"background={background.shape[1]}, overlay={overlay.shape[1]}"
        )
    mixed = np.concatenate([background, overlay], axis=0)
    if fusion_mode == "concat_sample":
        output_count = background.shape[0]
        if mixed.shape[0] >= output_count:
            sample_indices = rng.choice(mixed.shape[0], output_count, replace=False)
        else:
            sample_indices = rng.choice(mixed.shape[0], output_count, replace=True)
        mixed = mixed[sample_indices]
    return mixed[rng.permutation(mixed.shape[0])]


def predict_probabilities(model, mixed_points, args, device):
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(mixed_points), int(args.batch_size)):
            batch = np.stack(mixed_points[start:start + int(args.batch_size)], axis=0)
            tensor = torch.tensor(
                batch,
                dtype=torch.float32,
                device=device,
            ).transpose(2, 1)
            output = model(tensor)
            if isinstance(output, tuple):
                output = output[0]
            if args.entropy_mode == "sigmoid":
                prob = torch.sigmoid(output)
            else:
                prob = torch.softmax(output, dim=1)
            probabilities.append(prob.detach().cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def strip_entropy_for_sample(model, points, overlay_pool, args, device, rng):
    overlay_indices = rng.randint(0, len(overlay_pool), size=int(args.num_mix))
    mixed_points = [
        strip_fuse(points, overlay_pool[int(index)], rng, args.fusion_mode)
        for index in overlay_indices
    ]
    probabilities = predict_probabilities(model, mixed_points, args, device)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
    entropy = float(np.mean(entropy))
    if args.entropy_mode == "softmax_norm":
        num_classes = probabilities.shape[1]
        return entropy / np.log2(num_classes) * 100.0
    return entropy * 10.0


def compute_entropy_distribution(model, samples, overlay_pool, args, device, name, seed_offset):
    rng = np.random.RandomState(int(args.seed) + int(seed_offset))
    entropies = []
    for points in tqdm(samples, desc=f"STRIP {name}"):
        entropies.append(
            strip_entropy_for_sample(
                model,
                points,
                overlay_pool,
                args,
                device,
                rng,
            )
        )
    return np.asarray(entropies, dtype=np.float64)


def histogram_overlap(benign_entropy, attack_entropy, bins):
    lower = float(min(np.min(benign_entropy), np.min(attack_entropy)))
    upper = float(max(np.max(benign_entropy), np.max(attack_entropy)))
    if lower == upper:
        upper = lower + 1.0
    bin_edges = np.linspace(lower, upper, int(bins) + 1)
    benign_hist, _ = np.histogram(
        benign_entropy,
        bins=bin_edges,
        weights=np.ones_like(benign_entropy) / len(benign_entropy),
    )
    attack_hist, _ = np.histogram(
        attack_entropy,
        bins=bin_edges,
        weights=np.ones_like(attack_entropy) / len(attack_entropy),
    )
    return float(np.sum(np.minimum(benign_hist, attack_hist))), bin_edges


def auc_from_entropy(benign_entropy, attack_entropy):
    """AUROC for detecting low-entropy triggered samples."""
    scores = np.concatenate([-benign_entropy, -attack_entropy])
    labels = np.concatenate([
        np.zeros_like(benign_entropy, dtype=np.int32),
        np.ones_like(attack_entropy, dtype=np.int32),
    ])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    positive_ranks = ranks[labels == 1]
    num_pos = len(attack_entropy)
    num_neg = len(benign_entropy)
    if num_pos == 0 or num_neg == 0:
        return 0.0
    return float((np.sum(positive_ranks) - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


def auc_from_scores(benign_scores, attack_scores):
    scores = np.concatenate([benign_scores, attack_scores])
    labels = np.concatenate([
        np.zeros_like(benign_scores, dtype=np.int32),
        np.ones_like(attack_scores, dtype=np.int32),
    ])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    positive_ranks = ranks[labels == 1]
    num_pos = len(attack_scores)
    num_neg = len(benign_scores)
    if num_pos == 0 or num_neg == 0:
        return 0.0
    return float((np.sum(positive_ranks) - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


def save_entropy_csv(path, filenames, clean_labels, benign_entropy, attack_entropy):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "filename", "clean_label", "benign_entropy", "attack_entropy"])
        for index, (filename, label, benign_value, attack_value) in enumerate(
            zip(filenames, clean_labels, benign_entropy, attack_entropy)
        ):
            writer.writerow([
                index,
                filename,
                int(label),
                f"{float(benign_value):.8f}",
                f"{float(attack_value):.8f}",
            ])


def attack_display_name(attack_method):
    if attack_method == "regionba":
        return "RegionBA"
    if attack_method == "pointba_i":
        return "PointPBA-I"
    return attack_method


def nice_step(span, target_ticks=5):
    if span <= 0:
        return 1.0
    raw_step = span / max(int(target_ticks), 1)
    exponent = math.floor(math.log10(raw_step))
    base = 10.0 ** exponent
    fraction = raw_step / base
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * base


def entropy_tick_step(span, target_ticks=5, min_step=5.0):
    if span <= 0:
        return float(min_step)
    raw_step = span / max(int(target_ticks), 1)
    if raw_step <= min_step:
        return float(min_step)
    step = math.ceil(raw_step / 5.0) * 5.0
    return float(step)


def entropy_hist_axis(values, target_ticks=5, pad_ratio=0.06, nonnegative=False):
    values = np.asarray(values, dtype=np.float64)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        pad = 2.5
        lower -= pad
        upper += pad
    span = upper - lower
    pad = max(span * float(pad_ratio), 1.25)
    axis_lower = lower - pad
    axis_upper = upper + pad
    if nonnegative and axis_lower < 0:
        axis_lower = 0.0
    step = entropy_tick_step(axis_upper - axis_lower, target_ticks=target_ticks)
    first_tick = math.ceil(axis_lower / step) * step
    ticks = np.arange(first_tick, axis_upper + step * 0.5, step)
    return axis_lower, axis_upper, ticks


def probability_axis(max_height, target_ticks=5):
    y_max = max(float(max_height) * 1.12, 1e-12)
    step = nice_step(y_max, target_ticks=target_ticks)
    ticks = np.arange(0.0, y_max + step * 0.5, step)
    return y_max, ticks


def save_histogram(path, benign_entropy, attack_entropy, attack_method, bins, ylim):
    overlap, _ = histogram_overlap(benign_entropy, attack_entropy, bins)
    all_entropy = np.concatenate([benign_entropy, attack_entropy])
    x_min, x_max, x_ticks = entropy_hist_axis(
        all_entropy,
        target_ticks=5,
        pad_ratio=0.06,
        nonnegative=True,
    )
    plot_bin_edges = np.linspace(x_min, x_max, int(bins) + 1)

    plt.figure(figsize=(4.0, 2.35))
    weights_benign = np.ones_like(benign_entropy) / len(benign_entropy)
    weights_attack = np.ones_like(attack_entropy) / len(attack_entropy)
    plt.hist(
        benign_entropy,
        bins=plot_bin_edges,
        weights=weights_benign,
        alpha=0.9,
        label="Benign",
    )
    plt.hist(
        attack_entropy,
        bins=plot_bin_edges,
        weights=weights_attack,
        alpha=0.9,
        label=attack_display_name(attack_method),
    )
    plt.title("normalized entropy", fontsize=9)
    plt.ylabel("Probability (%)")
    plt.xlim(x_min, x_max)
    plt.xticks(x_ticks)
    if float(ylim) > 0:
        plt.ylim(0.0, float(ylim))
    else:
        benign_hist, _ = np.histogram(
            benign_entropy,
            bins=plot_bin_edges,
            weights=weights_benign,
        )
        attack_hist, _ = np.histogram(
            attack_entropy,
            bins=plot_bin_edges,
            weights=weights_attack,
        )
        max_height = float(max(np.max(benign_hist), np.max(attack_hist)))
        y_max, y_ticks = probability_axis(max_height, target_ticks=5)
        plt.ylim(0.0, y_max)
        plt.yticks(y_ticks)
    plt.tick_params(axis="both", labelsize=8)
    plt.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.25,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return overlap


def main():
    args = parse_args()
    configure_dataset_args(args)
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.attack_method == "regionba" and args.region_data_path is None:
        args.region_data_path = str(
            get_region_data_path(
                args.model,
                args.num_category,
                args.dataset,
                "test",
                root=args.region_data_root,
                num_regions=16,
            )
        )

    device = torch.device(
        "cpu"
        if args.use_cpu or not torch.cuda.is_available()
        else "cuda"
    )
    model = build_model(args, device)

    num_eval = None if int(args.num_eval) <= 0 else int(args.num_eval)
    if args.attack_method == "regionba":
        clean_points, clean_labels, filenames = collect_region_pkl_clean_samples(
            args,
            limit=num_eval,
        )
        attack_points, attack_labels, attack_filenames = collect_regionba_samples(
            args,
            limit=num_eval,
        )
        if len(attack_points) != len(clean_points):
            pair_count = min(len(attack_points), len(clean_points))
            clean_points = clean_points[:pair_count]
            clean_labels = clean_labels[:pair_count]
            filenames = filenames[:pair_count]
            attack_points = attack_points[:pair_count]
            attack_labels = attack_labels[:pair_count]
            attack_filenames = attack_filenames[:pair_count]
    elif args.attack_method == "pointba_i":
        clean_test_dataset = create_clean_dataset(args, split="test")
        clean_points, clean_labels, filenames = collect_clean_samples(
            clean_test_dataset,
            target_label=args.target_label,
            limit=num_eval,
            exclude_target=True,
        )
        attack_points, attack_labels = collect_pointba_i_samples(
            clean_points,
            clean_labels,
            args.target_label,
        )
    else:
        raise ValueError(f"Unsupported attack method: {args.attack_method}")

    overlay_dataset = create_clean_dataset(args, split=args.overlay_split)
    overlay_points, _, _ = collect_clean_samples(
        overlay_dataset,
        target_label=None,
        limit=None,
        exclude_target=False,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("visualization")
        / "strip"
        / f"{args.dataset}_{args.model}_{args.attack_method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== STRIP Defense Test ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Attack method: {args.attack_method}")
    print(f"Checkpoint: {args.model_path}")
    if args.attack_method == "regionba":
        print(f"Region PKL: {args.region_data_path}")
        print(f"Attack region mode: {args.attack_region_mode}")
        print(f"Grid density: {args.grid_density}")
    print(f"Target label: {args.target_label}")
    print(f"Evaluated samples: {len(clean_points)}")
    print(f"Benign overlays: {len(overlay_points)} ({args.overlay_split})")
    print(f"STRIP n_sample: {args.num_mix}")
    print(f"Preset FRR: {args.frr}")
    print(f"Fusion: {args.fusion_mode}")
    print(f"Entropy mode: {args.entropy_mode}")
    if args.entropy_mode == "softmax_norm":
        print("Entropy: mean softmax entropy over fused inputs, normalized to [0, 100]")
    else:
        print("Entropy: mean entropy over fused inputs, multiplied by 10 (legacy STRIP plot scale)")

    benign_entropy = compute_entropy_distribution(
        model,
        clean_points,
        overlay_points,
        args,
        device,
        "benign",
        seed_offset=0,
    )
    attack_entropy = compute_entropy_distribution(
        model,
        attack_points,
        overlay_points,
        args,
        device,
        args.attack_method,
        seed_offset=1000003,
    )

    lower_threshold = float(np.quantile(benign_entropy, float(args.frr)))
    lower_false_rejection_rate = float(np.mean(benign_entropy <= lower_threshold))
    lower_false_acceptance_rate = float(np.mean(attack_entropy > lower_threshold))
    lower_detection_rate = float(np.mean(attack_entropy <= lower_threshold))

    upper_threshold = float(np.quantile(benign_entropy, 1.0 - float(args.frr)))
    upper_false_rejection_rate = float(np.mean(benign_entropy >= upper_threshold))
    upper_false_acceptance_rate = float(np.mean(attack_entropy < upper_threshold))
    upper_detection_rate = float(np.mean(attack_entropy >= upper_threshold))

    half_frr = float(args.frr) / 2.0
    two_sided_lower = float(np.quantile(benign_entropy, half_frr))
    two_sided_upper = float(np.quantile(benign_entropy, 1.0 - half_frr))
    benign_two_sided_reject = (benign_entropy <= two_sided_lower) | (
        benign_entropy >= two_sided_upper
    )
    attack_two_sided_reject = (attack_entropy <= two_sided_lower) | (
        attack_entropy >= two_sided_upper
    )
    two_sided_false_rejection_rate = float(np.mean(benign_two_sided_reject))
    two_sided_false_acceptance_rate = float(np.mean(~attack_two_sided_reject))
    two_sided_detection_rate = float(np.mean(attack_two_sided_reject))

    overlap, _ = histogram_overlap(benign_entropy, attack_entropy, args.bins)
    low_entropy_auroc = auc_from_entropy(benign_entropy, attack_entropy)
    high_entropy_auroc = auc_from_scores(benign_entropy, attack_entropy)
    benign_mean = float(np.mean(benign_entropy))
    benign_std = float(np.std(benign_entropy))
    if benign_std < 1e-12:
        benign_std = 1.0
    benign_anomaly = np.abs((benign_entropy - benign_mean) / benign_std)
    attack_anomaly = np.abs((attack_entropy - benign_mean) / benign_std)
    two_sided_auroc = auc_from_scores(benign_anomaly, attack_anomaly)

    entropy_csv = output_dir / "strip_entropy.csv"
    summary_csv = output_dir / "strip_summary.csv"
    figure_path = output_dir / "strip_entropy_hist.png"

    save_entropy_csv(
        entropy_csv,
        filenames,
        clean_labels,
        benign_entropy,
        attack_entropy,
    )
    save_histogram(
        figure_path,
        benign_entropy,
        attack_entropy,
        args.attack_method,
        args.bins,
        args.ylim,
    )

    with open(summary_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["benign_mean_entropy", f"{float(np.mean(benign_entropy)):.8f}"])
        writer.writerow(["benign_std_entropy", f"{float(np.std(benign_entropy)):.8f}"])
        writer.writerow(["attack_mean_entropy", f"{float(np.mean(attack_entropy)):.8f}"])
        writer.writerow(["attack_std_entropy", f"{float(np.std(attack_entropy)):.8f}"])
        writer.writerow(["lower_threshold_at_frr", f"{lower_threshold:.8f}"])
        writer.writerow(["lower_false_rejection_rate", f"{lower_false_rejection_rate:.8f}"])
        writer.writerow(["lower_false_acceptance_rate", f"{lower_false_acceptance_rate:.8f}"])
        writer.writerow(["lower_detection_rate", f"{lower_detection_rate:.8f}"])
        writer.writerow(["upper_threshold_at_frr", f"{upper_threshold:.8f}"])
        writer.writerow(["upper_false_rejection_rate", f"{upper_false_rejection_rate:.8f}"])
        writer.writerow(["upper_false_acceptance_rate", f"{upper_false_acceptance_rate:.8f}"])
        writer.writerow(["upper_detection_rate", f"{upper_detection_rate:.8f}"])
        writer.writerow(["two_sided_lower_threshold", f"{two_sided_lower:.8f}"])
        writer.writerow(["two_sided_upper_threshold", f"{two_sided_upper:.8f}"])
        writer.writerow(["two_sided_false_rejection_rate", f"{two_sided_false_rejection_rate:.8f}"])
        writer.writerow(["two_sided_false_acceptance_rate", f"{two_sided_false_acceptance_rate:.8f}"])
        writer.writerow(["two_sided_detection_rate", f"{two_sided_detection_rate:.8f}"])
        writer.writerow(["histogram_overlap", f"{overlap:.8f}"])
        writer.writerow(["low_entropy_detection_auroc", f"{low_entropy_auroc:.8f}"])
        writer.writerow(["high_entropy_detection_auroc", f"{high_entropy_auroc:.8f}"])
        writer.writerow(["two_sided_detection_auroc", f"{two_sided_auroc:.8f}"])
        writer.writerow(["num_eval", len(clean_points)])
        writer.writerow(["num_mix", int(args.num_mix)])
        writer.writerow(["fusion_mode", args.fusion_mode])
        writer.writerow(["entropy_mode", args.entropy_mode])
        writer.writerow(["histogram_ylim", f"{float(args.ylim):.8f}"])

    print("\nMetric\tValue")
    print(f"Benign entropy mean/std\t{np.mean(benign_entropy):.4f}/{np.std(benign_entropy):.4f}")
    print(f"Attack entropy mean/std\t{np.mean(attack_entropy):.4f}/{np.std(attack_entropy):.4f}")
    print(f"Lower threshold @ FRR={args.frr}\t{lower_threshold:.4f}")
    print(f"Lower-tail false acceptance rate\t{lower_false_acceptance_rate:.6f}")
    print(f"Lower-tail detection rate\t{lower_detection_rate:.6f}")
    print(f"Upper threshold @ FRR={args.frr}\t{upper_threshold:.4f}")
    print(f"Upper-tail false acceptance rate\t{upper_false_acceptance_rate:.6f}")
    print(f"Upper-tail detection rate\t{upper_detection_rate:.6f}")
    print(f"Two-sided thresholds @ FRR={args.frr}\t{two_sided_lower:.4f}/{two_sided_upper:.4f}")
    print(f"Two-sided false acceptance rate\t{two_sided_false_acceptance_rate:.6f}")
    print(f"Two-sided detection rate\t{two_sided_detection_rate:.6f}")
    print(f"Histogram overlap\t{overlap:.6f}")
    print(f"Low-entropy detection AUROC\t{low_entropy_auroc:.6f}")
    print(f"High-entropy detection AUROC\t{high_entropy_auroc:.6f}")
    print(f"Two-sided detection AUROC\t{two_sided_auroc:.6f}")
    print(f"Entropy CSV: {entropy_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Histogram: {figure_path}")


if __name__ == "__main__":
    main()
