import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

from data_utils.attack_methods import pointba_o
from data_utils.dataset_config import (
    configure_dataset_args,
    create_backdoor_dataset,
    create_clean_dataset,
)
from model_config import get_region_data_path, import_model_module


DEFENSES = ["none", "rotation", "jitter", "scaling", "shifting", "all"]
ATTACK_METHODS = ["regionba", "pointba_o"]
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
    parser = argparse.ArgumentParser("Defense test for regional backdoor ASR")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--model", default="dgcnn",
                        help="model name: dgcnn, pointnet++, or curvenet")
    parser.add_argument("--model_path", type=str, required=True,
                        help="backdoored model checkpoint")
    parser.add_argument("--dataset", type=str, default="modelnet40",
                        help="dataset: modelnet10, modelnet40, or shapenetpart16")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_category", default=None, type=int, choices=[10, 16, 40])
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--use_uniform_sample", action="store_true", default=True)

    parser.add_argument("--target_label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--attack_method", type=str, default="regionba",
                        choices=ATTACK_METHODS)
    parser.add_argument("--grid_density", type=float, default=0.4)
    parser.add_argument("--attack_region_mode", type=str, default="top4",
                        choices=ATTACK_REGION_MODES)
    parser.add_argument("--attack_region_idx", type=int, default=None)

    parser.add_argument("--region_data_path", type=str, default=None,
                        help="test-set region PKL; inferred when omitted")
    parser.add_argument("--region_data_root", type=str, default="data")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--defenses", nargs="+", default=DEFENSES,
                        choices=DEFENSES,
                        help="defenses to test")

    parser.add_argument("--jitter_sigma", type=float, default=0.01)
    parser.add_argument("--jitter_clip", type=float, default=0.02)
    parser.add_argument("--scale_low", type=float, default=2.0 / 3.0)
    parser.add_argument("--scale_high", type=float, default=3.0 / 2.0)
    parser.add_argument("--shift_range", type=float, default=0.2)
    parser.add_argument("--rotation_axis", type=str, default="y",
                        choices=["x", "y", "z"])
    parser.add_argument("--save_perturbation_visualization",
                        action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def clean_state_dict_keys(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(args, device):
    model_module = import_model_module(args.model)
    model = model_module.get_model(args.num_category, normal_channel=args.use_normals)

    checkpoint = load_checkpoint(args.model_path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(clean_state_dict_keys(state_dict))
    model = model.to(device)
    model.eval()
    return model


class PointBAODefenseDataset(Dataset):
    def __init__(self, clean_dataset, target_label):
        self.clean_dataset = clean_dataset
        self.target_label = int(target_label)
        self.indices = [
            index
            for index in range(len(clean_dataset))
            if int(clean_dataset[index][1]) != self.target_label
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        points, label = self.clean_dataset[self.indices[index]]
        points = np.asarray(points, dtype=np.float32).copy()
        labels = np.asarray([[label]], dtype=np.int64)
        attack_points = points[None, :, :3].copy()
        pointba_o(attack_points, labels, frozenset([0]), self.target_label)
        points[:, :3] = attack_points[0]
        return points, int(labels[0, 0])


def rotation_matrix(angle, axis):
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    if axis == "x":
        return np.array([
            [1, 0, 0],
            [0, cosval, -sinval],
            [0, sinval, cosval],
        ], dtype=np.float32)
    if axis == "z":
        return np.array([
            [cosval, -sinval, 0],
            [sinval, cosval, 0],
            [0, 0, 1],
        ], dtype=np.float32)
    return np.array([
        [cosval, 0, sinval],
        [0, 1, 0],
        [-sinval, 0, cosval],
    ], dtype=np.float32)


def defense_rotation(batch_data, axis="y"):
    defended = np.asarray(batch_data, dtype=np.float32).copy()
    for batch_idx in range(defended.shape[0]):
        matrix = rotation_matrix(np.random.uniform() * 2.0 * np.pi, axis)
        defended[batch_idx, :, :3] = np.dot(defended[batch_idx, :, :3], matrix)
        if defended.shape[2] >= 6:
            defended[batch_idx, :, 3:6] = np.dot(defended[batch_idx, :, 3:6], matrix)
    return defended


def defense_jitter(batch_data, sigma=0.01, clip=0.05):
    defended = np.asarray(batch_data, dtype=np.float32).copy()
    noise = np.clip(
        sigma * np.random.randn(*defended[:, :, :3].shape),
        -clip,
        clip,
    ).astype(np.float32)
    defended[:, :, :3] += noise
    return defended


def defense_scaling(batch_data, scale_low=2.0 / 3.0, scale_high=3.0 / 2.0):
    defended = np.asarray(batch_data, dtype=np.float32).copy()
    scales = np.random.uniform(
        low=scale_low,
        high=scale_high,
        size=(defended.shape[0], 1, 3),
    ).astype(np.float32)
    defended[:, :, :3] *= scales
    return defended


def defense_shifting(batch_data, shift_range=0.2):
    defended = np.asarray(batch_data, dtype=np.float32).copy()
    shifts = np.random.uniform(
        low=-shift_range,
        high=shift_range,
        size=(defended.shape[0], 1, 3),
    ).astype(np.float32)
    defended[:, :, :3] += shifts
    return defended


def apply_defense(batch_data, defense_name, args):
    if defense_name == "none":
        return np.asarray(batch_data, dtype=np.float32)
    if defense_name == "rotation":
        return defense_rotation(batch_data, axis=args.rotation_axis)
    if defense_name == "jitter":
        return defense_jitter(
            batch_data,
            sigma=args.jitter_sigma,
            clip=args.jitter_clip,
        )
    if defense_name == "scaling":
        return defense_scaling(
            batch_data,
            scale_low=args.scale_low,
            scale_high=args.scale_high,
        )
    if defense_name == "shifting":
        return defense_shifting(batch_data, shift_range=args.shift_range)
    if defense_name == "all":
        defended = defense_rotation(batch_data, axis=args.rotation_axis)
        defended = defense_jitter(defended, sigma=args.jitter_sigma, clip=args.jitter_clip)
        defended = defense_scaling(defended, scale_low=args.scale_low, scale_high=args.scale_high)
        defended = defense_shifting(defended, shift_range=args.shift_range)
        return defended
    raise ValueError(f"Unsupported defense: {defense_name}")


def evaluate_asr(model, loader, defense_name, args, device):
    correct = 0
    total = 0
    progress = tqdm(enumerate(loader), total=len(loader), desc=defense_name)

    with torch.no_grad():
        for batch_idx, (points, target) in progress:
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            points = points.numpy()
            points = apply_defense(points, defense_name, args)
            points = torch.tensor(points, dtype=torch.float32).transpose(2, 1).to(device)
            target = target.long().to(device)

            output = model(points)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.data.max(1)[1]

            correct += pred.eq(target).sum().item()
            total += int(target.numel())

    return correct / float(total) if total > 0 else 0.0, total


def main():
    args = parse_args()
    configure_dataset_args(args)
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.attack_method == "regionba" and args.region_data_path is None:
        args.region_data_path = str(get_region_data_path(
            args.model,
            args.num_category,
            args.dataset,
            "test",
            root=args.region_data_root,
            num_regions=16,
        ))

    args.disable_perturbation_visualization = not args.save_perturbation_visualization

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    model = build_model(args, device)

    if args.attack_method == "regionba":
        test_bd_dataset = create_backdoor_dataset(args, split="test")
    elif args.attack_method == "pointba_o":
        clean_test_dataset = create_clean_dataset(args, split="test")
        test_bd_dataset = PointBAODefenseDataset(
            clean_test_dataset,
            target_label=args.target_label,
        )
    else:
        raise ValueError(f"Unsupported attack method: {args.attack_method}")

    test_bd_loader = DataLoader(
        test_bd_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("\n=== Defense ASR Test ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Attack method: {args.attack_method}")
    print(f"Checkpoint: {args.model_path}")
    if args.attack_method == "regionba":
        print(f"Region PKL: {args.region_data_path}")
    print(f"Target label: {args.target_label}")
    if args.attack_method == "regionba":
        print(f"Attack region mode: {args.attack_region_mode}")
        print(f"Grid density: {args.grid_density}")

    results = []
    for defense_name in args.defenses:
        set_seed(args.seed + DEFENSES.index(defense_name))
        asr, sample_count = evaluate_asr(
            model,
            test_bd_loader,
            defense_name,
            args,
            device,
        )
        results.append((defense_name, asr, sample_count))

    print("\nDefense\tASR\tSamples")
    for defense_name, asr, sample_count in results:
        print(f"{defense_name}\t{asr:.6f}\t{sample_count}")


if __name__ == "__main__":
    main()
