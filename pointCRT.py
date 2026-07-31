import argparse
import copy
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_utils.dataset_config import configure_dataset_args
from defense import build_model
from model_config import get_region_data_path
from strip import (
    collect_pointba_i_samples,
    collect_region_pkl_clean_samples,
    collect_regionba_samples,
    set_seed,
)


POINTCRT_CORRUPTIONS = [
    "background",
    "cutout",
    "density",
    "density_inc",
    "distortion",
    "distortion_rbf",
    "distortion_rbf_inv",
    "gaussian",
    "impulse",
    "rotation",
    "scale",
    "shear",
    "uniform",
    "upsampling",
    "ufsampling",
]


def parse_args():
    parser = argparse.ArgumentParser(
        "PointCRT defense test for ModelNet40 + DGCNN"
    )
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument("--model", type=str, default="dgcnn")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_category", type=int, default=None, choices=[40])
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--use_uniform_sample", action="store_true", default=True)

    parser.add_argument("--target_label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument(
        "--attack_methods",
        nargs="+",
        default=["regionba", "pointba_i"],
        choices=["regionba", "pointba_i"],
    )
    parser.add_argument("--regionba_model_path", type=str, default=None)
    parser.add_argument("--pointba_i_model_path", type=str, default=None)
    parser.add_argument("--attack_region_mode", type=str, default="top2")
    parser.add_argument("--attack_region_idx", type=int, default=None)
    parser.add_argument("--grid_density", type=float, default=0.4)
    parser.add_argument("--region_data_path", type=str, default=None)
    parser.add_argument("--region_data_root", type=str, default="data")

    parser.add_argument(
        "--num_eval",
        type=int,
        default=2000,
        help="number of non-target test samples used for clean/poisoned sets",
    )
    parser.add_argument(
        "--severity_max",
        type=int,
        default=5,
        help="PointCRT uses severity levels 1..5",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.9,
        help="test split ratio for the PointCRT detector; original code uses 0.9",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    return parser.parse_args()


def normalize_points(points):
    points = np.asarray(points, dtype=np.float32)
    centroid = np.mean(points[:, :3], axis=0, keepdims=True)
    output = points.copy()
    output[:, :3] = output[:, :3] - centroid
    radius = np.max(np.sqrt(np.sum(output[:, :3] ** 2, axis=1)))
    if radius > 1e-12:
        output[:, :3] = output[:, :3] / radius
    return output.astype(np.float32)


def random_rotation_matrix(rng, severity):
    degree = [10, 20, 30, 40, 50][severity - 1]
    angles = (
        rng.uniform(degree - 2.5, degree + 2.5, size=3)
        * rng.choice([-1.0, 1.0], size=3)
        * np.pi
        / 180.0
    )
    theta, gamma, beta = angles
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [np.cos(gamma), 0, np.sin(gamma)],
            [0, 1, 0],
            [-np.sin(gamma), 0, np.cos(gamma)],
        ],
        dtype=np.float32,
    )
    rz = np.array(
        [
            [np.cos(beta), -np.sin(beta), 0],
            [np.sin(beta), np.cos(beta), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return rx @ ry @ rz


def delete_neighborhood(pointcloud, count, neighbors, rng, keep_fraction=0.0):
    pointcloud = np.asarray(pointcloud, dtype=np.float32).copy()
    for _ in range(count):
        if len(pointcloud) <= neighbors + 1:
            break
        picked = pointcloud[rng.choice(len(pointcloud), 1)]
        dist = np.sum((pointcloud[:, :3] - picked[:, :3]) ** 2, axis=1)
        selected = np.argpartition(dist, neighbors)[:neighbors]
        if keep_fraction > 0:
            remove_count = max(1, int((1.0 - keep_fraction) * neighbors))
            selected = rng.choice(selected, remove_count, replace=False)
        pointcloud = np.delete(pointcloud, selected, axis=0)
    return pointcloud


def smooth_distortion(pointcloud, severity, rng, inverse=False):
    pointcloud = np.asarray(pointcloud, dtype=np.float32).copy()
    amplitude = [0.08, 0.16, 0.24, 0.32, 0.40][severity - 1]
    anchors = rng.uniform(-1.0, 1.0, size=(5, 3)).astype(np.float32)
    directions = rng.normal(size=(5, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    distances = np.sum(
        (pointcloud[:, None, :3] - anchors[None, :, :]) ** 2,
        axis=2,
    )
    if inverse:
        weights = 1.0 / (distances + 0.05)
    else:
        weights = np.exp(-distances / 0.35)
    weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-12)
    pointcloud[:, :3] += amplitude * (weights @ directions)
    return normalize_points(pointcloud)


def apply_pointcrt_corruption(pointcloud, corruption, severity, rng, origin_num=1024):
    pointcloud = np.asarray(pointcloud, dtype=np.float32).copy()
    pointcloud = pointcloud[:, :3]
    num_points, channels = pointcloud.shape

    if corruption == "uniform":
        scale = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        return normalize_points(pointcloud + rng.uniform(-scale, scale, pointcloud.shape))

    if corruption == "gaussian":
        scale = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        return np.clip(pointcloud + rng.normal(size=pointcloud.shape) * scale, -1, 1).astype(np.float32)

    if corruption == "background":
        count = [num_points // 50, num_points // 40, num_points // 30, num_points // 20, num_points // 10][severity - 1]
        background = rng.uniform(-1, 1, size=(count, channels)).astype(np.float32)
        return normalize_points(np.concatenate([pointcloud, background], axis=0))

    if corruption == "impulse":
        count = min([num_points // 5, num_points // 4, num_points // 3, num_points // 2, num_points][severity - 1], num_points)
        index = rng.choice(num_points, count, replace=False)
        pointcloud[index] += rng.choice([-1.0, 1.0], size=(count, channels)) * 0.1
        return normalize_points(pointcloud)

    if corruption == "rotation":
        return normalize_points(pointcloud @ random_rotation_matrix(rng, severity))

    if corruption == "shear":
        scale = [0.1, 0.3, 0.5, 0.7, 0.9][severity - 1]
        values = rng.uniform(scale - 0.05, scale + 0.05, size=6) * rng.choice([-1.0, 1.0], size=6)
        a, b, d, e, f, _ = values
        matrix = np.array([[1, 0, b], [d, 1, e], [f, 0, 1]], dtype=np.float32)
        return normalize_points(pointcloud @ matrix)

    if corruption == "scale":
        scale = [0.1, 0.3, 0.5, 0.7, 0.9][severity - 1]
        factors = np.ones(3, dtype=np.float32)
        axis = rng.randint(0, 3)
        sign = rng.choice([-1.0, 1.0])
        other = (axis + 1) % 3
        factors[axis] += scale * sign
        factors[other] -= scale * sign
        return normalize_points(pointcloud * factors)

    if corruption == "cutout":
        settings = [(10, 30), (15, 40), (15, 45), (18, 45), (16, 56)][severity - 1]
        return normalize_points(delete_neighborhood(pointcloud, settings[0], settings[1], rng))

    if corruption == "density":
        settings = [(1, 200), (2, 200), (3, 200), (4, 200), (5, 200)][severity - 1]
        return normalize_points(
            delete_neighborhood(
                pointcloud,
                settings[0],
                settings[1],
                rng,
                keep_fraction=0.25,
            )
        )

    if corruption == "density_inc":
        settings = [(1, 150), (3, 150), (4, 150), (4, 200), (5, 200)][severity - 1]
        remaining = pointcloud.copy()
        selected_parts = []
        for _ in range(settings[0]):
            if len(remaining) <= settings[1] + 1:
                break
            picked = remaining[rng.choice(len(remaining), 1)]
            dist = np.sum((remaining[:, :3] - picked[:, :3]) ** 2, axis=1)
            selected = np.argpartition(dist, settings[1])[:settings[1]]
            selected_parts.append(remaining[selected])
            remaining = np.delete(remaining, selected, axis=0)
        needed = max(origin_num - sum(len(part) for part in selected_parts), 1)
        replace = len(remaining) < needed
        fill = remaining[rng.choice(len(remaining), needed, replace=replace)]
        selected_parts.append(fill)
        return normalize_points(np.concatenate(selected_parts, axis=0))

    if corruption == "upsampling":
        count = [num_points // 5, num_points // 4, num_points // 3, num_points // 2, num_points][severity - 1]
        index = rng.choice(num_points, min(count, num_points), replace=False)
        added = pointcloud[index] + rng.uniform(-0.1, 0.1, size=(len(index), channels))
        return normalize_points(np.concatenate([pointcloud, added.astype(np.float32)], axis=0))

    if corruption == "ufsampling":
        remove_count = [200, 400, 600, 800, 896][severity - 1]
        keep_count = max(num_points - remove_count, 1)
        index = rng.choice(num_points, keep_count, replace=False)
        return normalize_points(pointcloud[index])

    if corruption == "distortion":
        return smooth_distortion(pointcloud, severity, rng, inverse=False)

    if corruption == "distortion_rbf":
        return smooth_distortion(pointcloud, severity, rng, inverse=False)

    if corruption == "distortion_rbf_inv":
        return smooth_distortion(pointcloud, severity, rng, inverse=True)

    raise ValueError(f"Unsupported corruption: {corruption}")


def get_model_path(args, attack_method):
    if attack_method == "regionba":
        return args.regionba_model_path
    if attack_method == "pointba_i":
        return args.pointba_i_model_path
    raise ValueError(f"Unsupported attack method: {attack_method}")


def predict_point_clouds(model, samples, args, device, desc):
    predictions = np.full(len(samples), -1, dtype=np.int64)
    grouped_indices = {}
    for index, points in enumerate(samples):
        points = np.asarray(points, dtype=np.float32)[:, :3]
        grouped_indices.setdefault(points.shape, []).append(index)

    with torch.no_grad():
        for shape, indices in grouped_indices.items():
            for start in tqdm(
                range(0, len(indices), int(args.batch_size)),
                desc=f"{desc} N={shape[0]}",
                leave=False,
            ):
                batch_indices = indices[start:start + int(args.batch_size)]
                batch = np.stack(
                    [
                        np.asarray(samples[index], dtype=np.float32)[:, :3]
                        for index in batch_indices
                    ],
                    axis=0,
                )
                tensor = torch.tensor(
                    batch,
                    dtype=torch.float32,
                    device=device,
                ).transpose(2, 1)
                output = model(tensor)
                if isinstance(output, tuple):
                    output = output[0]
                pred = output.argmax(dim=1).detach().cpu().numpy()
                predictions[batch_indices] = pred
    return predictions


def compute_pointcrt_features(model, samples, original_predictions, args, device, name, seed_offset):
    samples = [np.asarray(points, dtype=np.float32)[:, :3].copy() for points in samples]
    features = np.full(
        (len(samples), len(POINTCRT_CORRUPTIONS)),
        int(args.severity_max) + 1,
        dtype=np.int64,
    )

    progress = tqdm(
        POINTCRT_CORRUPTIONS,
        desc=f"PointCRT features ({name})",
    )
    for corruption_index, corruption in enumerate(progress):
        progress.set_postfix_str(corruption)
        for severity in range(1, int(args.severity_max) + 1):
            rng = np.random.RandomState(
                int(args.seed)
                + int(seed_offset)
                + corruption_index * 1009
                + severity * 9173
            )
            corrupted = [
                apply_pointcrt_corruption(
                    points,
                    corruption,
                    severity,
                    rng,
                    origin_num=int(args.num_point),
                )
                for points in samples
            ]
            predictions = predict_point_clouds(
                model,
                corrupted,
                args,
                device,
                desc=f"{name}:{corruption}:s{severity}",
            )
            unchanged = features[:, corruption_index] == int(args.severity_max) + 1
            changed = predictions != original_predictions
            features[unchanged & changed, corruption_index] = severity

    return features


def train_pointcrt_detector(clean_features, attack_features, args):
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    classifier_name = "xgboost"
    try:
        import xgboost

        classifier = xgboost.XGBClassifier(
            learning_rate=0.05,
            n_estimators=100,
            max_depth=5,
            seed=2023,
            subsample=0.8,
            colsample_bytree=0.7,
            n_jobs=8,
            eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        classifier_name = "random_forest_fallback"
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=2023,
            n_jobs=8,
        )

    inputs = np.concatenate([clean_features, attack_features], axis=0)
    labels = np.concatenate(
        [
            np.zeros(len(clean_features), dtype=np.int64),
            np.ones(len(attack_features), dtype=np.int64),
        ],
        axis=0,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        labels,
        test_size=float(args.test_ratio),
        stratify=labels,
        random_state=2023,
    )
    classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(x_test)[:, 1]
    else:
        scores = predicted.astype(np.float32)
    f1 = float(metrics.f1_score(y_test, predicted))
    auc = float(metrics.roc_auc_score(y_test, scores))
    return classifier_name, f1, auc


def save_features(path, filenames, clean_labels, original_predictions, features, sample_type):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["sample_type", "filename", "clean_label", "original_prediction"]
            + [f"{corruption}_first_changed_severity" for corruption in POINTCRT_CORRUPTIONS]
        )
        for index, filename in enumerate(filenames):
            writer.writerow(
                [
                    sample_type,
                    filename,
                    int(clean_labels[index]),
                    int(original_predictions[index]),
                ]
                + [int(value) for value in features[index]]
            )


def build_attack_samples(args, attack_method, clean_points, clean_labels):
    if attack_method == "regionba":
        attack_points, attack_labels, _ = collect_regionba_samples(
            args,
            limit=len(clean_points),
        )
        return attack_points, attack_labels
    if attack_method == "pointba_i":
        return collect_pointba_i_samples(
            clean_points,
            clean_labels,
            args.target_label,
        )
    raise ValueError(f"Unsupported attack method: {attack_method}")


def run_one_attack(args, attack_method, clean_points, clean_labels, filenames, device, output_root):
    model_path = get_model_path(args, attack_method)
    if not model_path:
        raise ValueError(f"--{attack_method}_model_path is required")

    method_args = copy.copy(args)
    method_args.model_path = model_path
    method_args.attack_method = attack_method
    model = build_model(method_args, device)

    attack_points, attack_labels = build_attack_samples(
        method_args,
        attack_method,
        clean_points,
        clean_labels,
    )
    pair_count = min(len(clean_points), len(attack_points))
    clean_points = clean_points[:pair_count]
    clean_labels = clean_labels[:pair_count]
    filenames = filenames[:pair_count]
    attack_points = attack_points[:pair_count]

    original_clean_pred = predict_point_clouds(
        model,
        clean_points,
        method_args,
        device,
        desc=f"{attack_method}:original_clean",
    )
    original_attack_pred = predict_point_clouds(
        model,
        attack_points,
        method_args,
        device,
        desc=f"{attack_method}:original_attack",
    )
    clean_acc = float(np.mean(original_clean_pred == np.asarray(clean_labels, dtype=np.int64)))
    attack_asr = float(np.mean(original_attack_pred == int(args.target_label)))

    clean_features = compute_pointcrt_features(
        model,
        clean_points,
        original_clean_pred,
        method_args,
        device,
        name=f"{attack_method}:clean",
        seed_offset=0,
    )
    attack_features = compute_pointcrt_features(
        model,
        attack_points,
        original_attack_pred,
        method_args,
        device,
        name=f"{attack_method}:attack",
        seed_offset=1000003,
    )
    classifier_name, f1, auc = train_pointcrt_detector(
        clean_features,
        attack_features,
        method_args,
    )

    method_dir = output_root / attack_method
    method_dir.mkdir(parents=True, exist_ok=True)
    save_features(
        method_dir / "clean_features.csv",
        filenames,
        clean_labels,
        original_clean_pred,
        clean_features,
        sample_type="clean",
    )
    save_features(
        method_dir / "attack_features.csv",
        filenames,
        clean_labels,
        original_attack_pred,
        attack_features,
        sample_type="attack",
    )
    summary_path = method_dir / "pointcrt_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["dataset", args.dataset])
        writer.writerow(["model", args.model])
        writer.writerow(["attack_method", attack_method])
        writer.writerow(["model_path", model_path])
        writer.writerow(["target_label", int(args.target_label)])
        writer.writerow(["num_eval", pair_count])
        writer.writerow(["clean_acc", f"{clean_acc:.8f}"])
        writer.writerow(["attack_asr", f"{attack_asr:.8f}"])
        writer.writerow(["classifier", classifier_name])
        writer.writerow(["f1", f"{f1:.8f}"])
        writer.writerow(["auc", f"{auc:.8f}"])
        writer.writerow(["test_ratio", f"{float(args.test_ratio):.8f}"])
        writer.writerow(["corruptions", " ".join(POINTCRT_CORRUPTIONS)])

    print(f"\n[{attack_method}]")
    print(f"Clean ACC: {clean_acc:.6f}")
    print(f"Attack ASR: {attack_asr:.6f}")
    print(f"PointCRT classifier: {classifier_name}")
    print(f"F1: {f1:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"Summary CSV: {summary_path}")

    return {
        "attack_method": attack_method,
        "clean_acc": clean_acc,
        "attack_asr": attack_asr,
        "classifier": classifier_name,
        "f1": f1,
        "auc": auc,
        "summary_path": str(summary_path),
    }


def main():
    args = parse_args()
    configure_dataset_args(args)
    if args.dataset != "modelnet40" or args.model.lower() != "dgcnn":
        raise ValueError("This first PointCRT implementation supports only ModelNet40 + DGCNN.")

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.region_data_path is None:
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
    num_eval = None if int(args.num_eval) <= 0 else int(args.num_eval)
    clean_points, clean_labels, filenames = collect_region_pkl_clean_samples(
        args,
        limit=num_eval,
    )

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("visualization") / "pointcrt" / f"{args.dataset}_{args.model}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n=== PointCRT Defense Test ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Region PKL: {args.region_data_path}")
    print(f"Target label: {args.target_label}")
    print(f"Evaluated non-target samples: {len(clean_points)}")
    print(f"Corruptions: {' '.join(POINTCRT_CORRUPTIONS)}")
    print(f"Severity levels: 1..{args.severity_max}")
    print(f"Output directory: {output_root}")

    summaries = []
    for attack_method in args.attack_methods:
        summaries.append(
            run_one_attack(
                args,
                attack_method,
                clean_points,
                clean_labels,
                filenames,
                device,
                output_root,
            )
        )

    combined_summary = output_root / "pointcrt_summary.csv"
    with open(combined_summary, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "attack_method",
            "clean_acc",
            "attack_asr",
            "classifier",
            "f1",
            "auc",
            "summary_path",
        ])
        for row in summaries:
            writer.writerow([
                row["attack_method"],
                f"{row['clean_acc']:.8f}",
                f"{row['attack_asr']:.8f}",
                row["classifier"],
                f"{row['f1']:.8f}",
                f"{row['auc']:.8f}",
                row["summary_path"],
            ])
    print(f"\nCombined summary CSV: {combined_summary}")


if __name__ == "__main__":
    main()
