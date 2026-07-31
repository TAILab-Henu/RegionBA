import argparse
import csv
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "models"))

from data_utils.ModelNetDataLoader import augment_point_cloud
from data_utils.dataset_config import configure_dataset_args, create_clean_dataset
from data_utils.ibapc.GFT import GFT_opt
from data_utils.ibapc.spectral_attack import eig_vector
from model_config import build_training_policy, get_model_storage_name, import_model_module


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, *rest = self.dataset[index]
        return index, points, label


def parse_args():
    parser = argparse.ArgumentParser(
        "Train IBAPC GFT noise with RegionBA data/model pipeline"
    )
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument(
        "--spectral_batch_size",
        type=int,
        default=2,
        help="chunk size for graph eigen decomposition; lower it if CUDA OOM",
    )
    parser.add_argument(
        "--model",
        default="dgcnn",
        help="victim model: dgcnn, pointnet++, or curvenet",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="modelnet40",
        help="dataset: modelnet40 or shapenetpart16",
    )
    parser.add_argument("--num_category", default=None, type=int, choices=[10, 16, 40])
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--attack_lr", default=0.01, type=float)
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--use_uniform_sample", action="store_true", default=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="log")
    parser.add_argument("--target_label", type=int, default=2)
    parser.add_argument("--poisoned_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--ibapc_knn", type=int, default=10)
    parser.add_argument(
        "--eigen_cache_root",
        type=str,
        default=None,
        help=(
            "root directory for cached IBAPC eigenvectors; if omitted, "
            "use /root/shared-nvme/eigen_cache when available"
        ),
    )
    parser.add_argument("--initial_noise_level", type=float, default=0.1)
    parser.add_argument("--poison_weight_attack", type=float, default=0.9)
    parser.add_argument("--poison_weight_dis", type=float, default=0.1)
    parser.add_argument(
        "--trigger_update_scope",
        type=str,
        default="batch",
        choices=["batch", "poison"],
        help="batch follows IBAPC by updating GFT noise on the whole training batch; poison is a faster approximation",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="deprecated; evaluation is performed only in the final 10 epochs",
    )
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        default=False,
        help="disable training augmentation for paper-code sanity checks",
    )
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def model_forward(model, points_bcn):
    output = model(points_bcn)
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def criterion_forward(criterion, logits, labels, trans_feat=None):
    try:
        return criterion(logits, labels.long(), trans_feat)
    except TypeError:
        return criterion(logits, labels.long())


def apply_ibapc_trigger(points_bnc, gft_noise, knn, spectral_batch_size):
    """Apply IBAPC trigger to BxNxC points, transforming only xyz channels."""
    if points_bnc.numel() == 0:
        return points_bnc

    poisoned_chunks = []
    for start in range(0, points_bnc.shape[0], spectral_batch_size):
        end = min(start + spectral_batch_size, points_bnc.shape[0])
        chunk = points_bnc[start:end]
        xyz = chunk[:, :, :3].contiguous()
        v, _, _ = eig_vector(xyz, knn)
        poisoned_xyz = GFT_opt(xyz, gft_noise, v)
        if chunk.shape[2] > 3:
            poisoned = chunk.clone()
            poisoned[:, :, :3] = poisoned_xyz
        else:
            poisoned = poisoned_xyz
        poisoned_chunks.append(poisoned)
    return torch.cat(poisoned_chunks, dim=0)


def apply_ibapc_trigger_with_v(points_bnc, gft_noise, v):
    """Apply IBAPC trigger with cached graph eigenvectors."""
    if points_bnc.numel() == 0:
        return points_bnc

    xyz = points_bnc[:, :, :3].contiguous()
    poisoned_xyz = GFT_opt(xyz, gft_noise, v)
    if points_bnc.shape[2] > 3:
        poisoned = points_bnc.clone()
        poisoned[:, :, :3] = poisoned_xyz
        return poisoned
    return poisoned_xyz


def build_poison_set(dataset, target_label, poisoned_rate, seed):
    labels = np.asarray([int(dataset[index][1]) for index in range(len(dataset))])
    candidates = np.where(labels != int(target_label))[0].tolist()
    rng = random.Random(seed)
    rng.shuffle(candidates)
    poison_num = min(int(len(labels) * float(poisoned_rate)), len(candidates))
    return frozenset(candidates[:poison_num]), labels


def get_train_eigen_cache_dir(args):
    sample_tag = "fps" if getattr(args, "use_uniform_sample", False) else "normal"
    cache_root = resolve_eigen_cache_root(args)

    cache_dir = cache_root / args.dataset / f"train_n{args.num_point}_{sample_tag}_knn{args.ibapc_knn}"
    if args.trigger_update_scope == "poison":
        rate_tag = str(args.poisoned_rate).replace(".", "p")
        cache_dir = cache_dir / f"target{args.target_label}_rate{rate_tag}_seed{args.seed}"
    return cache_dir


def get_test_eigen_cache_dir(args):
    sample_tag = "fps" if getattr(args, "use_uniform_sample", False) else "normal"
    return (
        resolve_eigen_cache_root(args)
        / args.dataset
        / f"test_n{args.num_point}_{sample_tag}_knn{args.ibapc_knn}"
    )


def resolve_eigen_cache_root(args):
    default_shared_cache_root = Path("/root/shared-nvme/eigen_cache")
    if getattr(args, "eigen_cache_root", None):
        return Path(args.eigen_cache_root)
    if default_shared_cache_root.exists():
        return default_shared_cache_root
    return ROOT_DIR / "data_utils" / "ibapc" / "eigen_cache"


def cached_eigen_path(cache_dir, sample_index):
    return cache_dir / f"{int(sample_index)}.npy"


def cached_eigen_is_valid(path, num_point):
    if not path.is_file():
        return False
    try:
        cached = np.load(path, mmap_mode="r")
        return cached.shape == (int(num_point), int(num_point))
    except Exception:
        return False


@torch.no_grad()
def precompute_poison_eigenvectors(dataset, poison_set, args, device, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_indices = [
        index
        for index in sorted(poison_set)
        if not cached_eigen_is_valid(cached_eigen_path(cache_dir, index), args.num_point)
    ]
    print("Poison eigen cache:", cache_dir)
    print(
        "Cached poison spectra:",
        len(poison_set) - len(missing_indices),
        "/",
        len(poison_set),
    )
    if not missing_indices:
        return

    for index in tqdm(missing_indices, desc="Precomputing poison spectra"):
        points, _ = dataset[index]
        xyz = np.asarray(points[:, :3], dtype=np.float32)
        point_tensor = torch.from_numpy(xyz).unsqueeze(0).to(device)
        v, _, _ = eig_vector(point_tensor, int(args.ibapc_knn))
        np.save(cached_eigen_path(cache_dir, index), v[0].cpu().numpy())


def load_cached_eigenvectors(indices, cache_dir, device, dtype):
    eigenvectors = [
        np.load(cached_eigen_path(cache_dir, index)).astype(np.float32, copy=False)
        for index in indices
    ]
    eigenvectors = np.stack(eigenvectors, axis=0)
    return torch.from_numpy(eigenvectors).to(device=device, dtype=dtype)


@torch.no_grad()
def load_or_compute_cached_eigenvectors(indices, points_bnc, cache_dir, args, device, dtype):
    eigenvectors = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for local_index, sample_index in enumerate(indices):
        path = cached_eigen_path(cache_dir, sample_index)
        if cached_eigen_is_valid(path, args.num_point):
            eigenvector = np.load(path).astype(np.float32, copy=False)
        else:
            point_tensor = points_bnc[local_index : local_index + 1, :, :3].detach()
            v, _, _ = eig_vector(point_tensor, int(args.ibapc_knn))
            eigenvector = v[0].cpu().numpy().astype(np.float32, copy=False)
            np.save(path, eigenvector)
        eigenvectors.append(eigenvector)
    eigenvectors = np.stack(eigenvectors, axis=0)
    return torch.from_numpy(eigenvectors).to(device=device, dtype=dtype)


def maybe_augment(points_bnc, use_augmentation):
    if not use_augmentation:
        return points_bnc
    augmented = augment_point_cloud(points_bnc.detach().cpu().numpy())
    return torch.from_numpy(augmented).to(points_bnc.device, dtype=points_bnc.dtype)


@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for _, points, labels in tqdm(loader, desc="Eval clean", leave=False):
        points = points.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device).long().view(-1)
        logits, _ = model_forward(model, points.transpose(2, 1).contiguous())
        pred = logits.argmax(dim=1)
        total += labels.numel()
        correct += pred.eq(labels).sum().item()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_asr(
    model,
    loader,
    gft_noise,
    args,
    device,
    test_eigen_cache_dir,
    exclude_target=True,
):
    model.eval()
    total = 0
    correct = 0
    desc = "Eval ASR non-target" if exclude_target else "Eval ASR all-test"
    for indices, points, labels in tqdm(loader, desc=desc, leave=False):
        indices = indices.numpy().tolist()
        labels = labels.long().view(-1)
        if exclude_target:
            keep = labels != int(args.target_label)
            if not torch.any(keep):
                continue
            keep_list = keep.numpy().tolist()
            indices = [index for index, should_keep in zip(indices, keep_list) if should_keep]
            points = points[keep]
        points = points.to(device=device, dtype=torch.float32)
        v = load_or_compute_cached_eigenvectors(
            indices,
            points,
            test_eigen_cache_dir,
            args,
            device,
            points.dtype,
        )
        target = torch.full(
            (points.shape[0],),
            int(args.target_label),
            device=device,
            dtype=torch.long,
        )
        poisoned = apply_ibapc_trigger_with_v(
            points,
            gft_noise,
            v,
        )
        logits, _ = model_forward(model, poisoned.transpose(2, 1).contiguous())
        pred = logits.argmax(dim=1)
        total += target.numel()
        correct += pred.eq(target).sum().item()
    return correct / max(total, 1)


def make_output_dirs(args):
    model_tag = get_model_storage_name(args.model)
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = (
        Path(args.output_root)
        / f"ibapc_{args.dataset}_{model_tag}"
        / f"target{args.target_label}_rate{args.poisoned_rate}_seed{args.seed}"
        / timestr
    )
    ckpt_dir = exp_dir / "checkpoints"
    noise_dir = exp_dir / "noises"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    canonical_dir = (
        ROOT_DIR
        / "data_utils"
        / "ibapc"
        / "noises"
        / args.dataset
        / model_tag
        / f"target{args.target_label}_seed{args.seed}"
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, ckpt_dir, noise_dir, canonical_dir


def save_checkpoint(path, model, gft_noise, epoch, clean_acc, asr, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "gft_noise": gft_noise.detach().cpu().numpy(),
            "clean_accuracy": clean_acc,
            "attack_success_rate": asr,
            "dataset": args.dataset,
            "model": args.model,
            "target_label": args.target_label,
            "poisoned_rate": args.poisoned_rate,
            "ibapc_knn": args.ibapc_knn,
        },
        path,
    )


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    configure_dataset_args(args)
    set_random_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    )
    use_augmentation = not args.no_augmentation

    exp_dir, ckpt_dir, noise_dir, canonical_dir = make_output_dirs(args)
    print("Output directory:", exp_dir)
    print("Canonical noise directory:", canonical_dir)
    print("Device:", device)

    with open(exp_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    print("Loading datasets...")
    train_dataset = create_clean_dataset(args, split="train", process_data=True)
    test_dataset = create_clean_dataset(args, split="test", process_data=True)
    poison_set, train_labels = build_poison_set(
        train_dataset,
        args.target_label,
        args.poisoned_rate,
        args.seed,
    )
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Target label:", args.target_label)
    print("Poisoned train samples:", len(poison_set))
    print("Trigger update scope:", args.trigger_update_scope)

    train_eigen_cache_dir = get_train_eigen_cache_dir(args)
    test_eigen_cache_dir = get_test_eigen_cache_dir(args)
    if args.trigger_update_scope == "poison":
        precompute_poison_eigenvectors(
            train_dataset,
            poison_set,
            args,
            device,
            train_eigen_cache_dir,
        )
    else:
        train_eigen_cache_dir.mkdir(parents=True, exist_ok=True)
        print("Train eigen cache:", train_eigen_cache_dir)
        print("Missing spectra will be cached on demand in the first epoch.")
    test_eigen_cache_dir.mkdir(parents=True, exist_ok=True)
    print("Test eigen cache:", test_eigen_cache_dir)
    print("Missing test spectra will be cached on demand during the first ASR evaluation.")

    train_loader = DataLoader(
        IndexedDataset(train_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        IndexedDataset(test_dataset),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    model_module = import_model_module(args.model)
    model = model_module.get_model(args.num_category, normal_channel=args.use_normals)
    criterion = model_module.get_loss()
    model = model.to(device)
    criterion = criterion.to(device)

    optimizer, scheduler, policy = build_training_policy(
        args.model,
        model.parameters(),
        learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        epochs=args.epoch,
        momentum=args.momentum,
    )
    print(
        "Optimizer:",
        policy["optimizer"],
        "effective lr:",
        policy["effective_learning_rate"],
        "scheduler:",
        policy["scheduler"],
    )
    print("Training augmentation:", "on" if use_augmentation else "off")

    gft_noise_np = (
        np.random.uniform(-0.5, 0.5, size=(args.num_point, 3)).astype(np.float32)
        * float(args.initial_noise_level)
    )
    gft_noise = torch.nn.Parameter(torch.tensor(gft_noise_np, device=device))
    trigger_optimizer = torch.optim.Adam(
        [gft_noise],
        lr=float(args.attack_lr),
        weight_decay=float(args.decay_rate),
    )
    trigger_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trigger_optimizer,
        T_max=int(args.epoch),
        eta_min=float(args.learning_rate),
    )
    print(
        "Trigger optimizer: Adam",
        "effective lr:",
        args.attack_lr,
        "scheduler: CosineAnnealingLR",
        "eta_min:",
        args.learning_rate,
    )

    metrics_path = exp_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_acc",
            "clean_acc",
            "asr_non_target",
            "avg_trigger_l2_loss_term",
        ])

    best_asr = -1.0
    best_clean_acc = -1.0

    for epoch in range(1, args.epoch + 1):
        print(f"\nEpoch {epoch}/{args.epoch}")
        model.train()
        if not policy["scheduler_step_after_epoch"]:
            scheduler.step()

        total = 0
        correct = 0
        trigger_l2_values = []

        for _, (indices, points, labels) in tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            smoothing=0.9,
        ):
            indices = indices.numpy().tolist()
            labels = labels.to(device=device).long().view(-1)
            points = points.to(device=device, dtype=torch.float32)

            poison_mask_cpu = np.asarray(
                [index in poison_set for index in indices],
                dtype=bool,
            )
            poison_mask = torch.from_numpy(poison_mask_cpu).to(device)
            poison_indices = [
                index
                for index, is_poison in zip(indices, poison_mask_cpu)
                if is_poison
            ]

            if args.trigger_update_scope == "batch":
                trigger_points = points
                trigger_v = load_or_compute_cached_eigenvectors(
                    indices,
                    trigger_points,
                    train_eigen_cache_dir,
                    args,
                    device,
                    trigger_points.dtype,
                )
            elif torch.any(poison_mask):
                trigger_points = points[poison_mask]
                trigger_v = load_cached_eigenvectors(
                    poison_indices,
                    train_eigen_cache_dir,
                    device,
                    trigger_points.dtype,
                )
            else:
                trigger_points = None
                trigger_v = None

            if trigger_points is not None:
                for param in model.parameters():
                    param.requires_grad_(False)
                model.eval()
                trigger_optimizer.zero_grad()
                poisoned_for_trigger = apply_ibapc_trigger_with_v(
                    trigger_points,
                    gft_noise,
                    trigger_v,
                )
                trigger_labels = torch.full(
                    (poisoned_for_trigger.shape[0],),
                    int(args.target_label),
                    device=device,
                    dtype=torch.long,
                )
                trigger_logits, trigger_feat = model_forward(
                    model,
                    poisoned_for_trigger.transpose(2, 1).contiguous(),
                )
                attack_loss = criterion_forward(
                    criterion,
                    trigger_logits,
                    trigger_labels,
                    trigger_feat,
                )
                l2_per_sample = torch.norm(
                    (
                        poisoned_for_trigger[:, :, :3]
                        - trigger_points[:, :, :3]
                    ).reshape(poisoned_for_trigger.shape[0], -1),
                    p=2,
                    dim=1,
                )
                l2_loss = torch.sum(l2_per_sample)
                trigger_loss = (
                    float(args.poison_weight_attack) * attack_loss
                    + float(args.poison_weight_dis) * l2_loss
                )
                trigger_loss.backward()
                trigger_optimizer.step()
                trigger_l2_values.append(float(torch.mean(l2_per_sample).detach().cpu()))

                for param in model.parameters():
                    param.requires_grad_(True)

            if torch.any(poison_mask):
                if args.trigger_update_scope == "batch":
                    poison_v = trigger_v[poison_mask]
                else:
                    poison_v = trigger_v
                with torch.no_grad():
                    points[poison_mask] = apply_ibapc_trigger_with_v(
                        points[poison_mask],
                        gft_noise,
                        poison_v,
                    )
                    labels[poison_mask] = int(args.target_label)

            points = maybe_augment(points, use_augmentation)

            model.train()
            optimizer.zero_grad()
            logits, trans_feat = model_forward(model, points.transpose(2, 1).contiguous())
            loss = criterion_forward(criterion, logits, labels, trans_feat)
            loss.backward()
            if policy["gradient_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    policy["gradient_clip_norm"],
                )
            optimizer.step()

            pred = logits.argmax(dim=1)
            total += labels.numel()
            correct += pred.eq(labels).sum().item()

        if policy["scheduler_step_after_epoch"]:
            scheduler.step()
        trigger_scheduler.step()

        train_acc = correct / max(total, 1)
        avg_trigger_l2 = float(np.mean(trigger_l2_values)) if trigger_l2_values else 0.0
        print(f"Train Accuracy: {train_acc:.6f}")
        print(f"Average trigger L2 loss term (not paper metric): {avg_trigger_l2:.6f}")

        should_asr_eval = epoch > max(int(args.epoch) - 10, 0)
        should_clean_eval = should_asr_eval
        if should_clean_eval:
            clean_acc = evaluate_clean(model, test_loader, device)
            print(f"Clean Test Instance Accuracy (BAc): {clean_acc:.6f}")
        else:
            clean_acc = float("nan")

        if should_asr_eval:
            asr = evaluate_asr(
                model,
                test_loader,
                gft_noise,
                args,
                device,
                test_eigen_cache_dir,
                exclude_target=True,
            )
            print(f"Backdoor Test Instance Accuracy (ASR, non-target): {asr:.6f}")
        else:
            asr = float("nan")

        with open(metrics_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_acc, clean_acc, asr, avg_trigger_l2])

        if should_clean_eval and clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
            save_checkpoint(
                ckpt_dir / "best_clean_model.pth",
                model,
                gft_noise,
                epoch,
                clean_acc,
                asr,
                args,
            )

        if should_asr_eval and asr > best_asr:
            best_asr = asr
            np.save(noise_dir / "GFT_noise_best_asr.npy", gft_noise.detach().cpu().numpy())
            save_checkpoint(
                ckpt_dir / "best_asr_model.pth",
                model,
                gft_noise,
                epoch,
                clean_acc,
                asr,
                args,
            )

        if epoch % max(args.save_every, 1) == 0 or epoch == args.epoch:
            np.save(
                noise_dir / f"GFT_noise_epoch{epoch}.npy",
                gft_noise.detach().cpu().numpy(),
            )

    final_noise = gft_noise.detach().cpu().numpy()
    np.save(noise_dir / "GFT_noise_final.npy", final_noise)
    np.save(canonical_dir / "GFT_noise.npy", final_noise)
    save_checkpoint(
        ckpt_dir / "final_model.pth",
        model,
        gft_noise,
        args.epoch,
        clean_acc,
        asr,
        args,
    )

    print("\nTraining completed.")
    print("Final noise:", noise_dir / "GFT_noise_final.npy")
    print("Canonical noise:", canonical_dir / "GFT_noise.npy")
    print("Final model:", ckpt_dir / "final_model.pth")
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main(parse_args())
