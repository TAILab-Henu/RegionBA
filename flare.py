import argparse
import copy
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_utils.dataset_config import configure_dataset_args, create_backdoor_dataset
from defense import build_model
from model_config import get_region_data_path

def parse_args():
    parser = argparse.ArgumentParser(
        "FLARE detection for poisoned training samples"
    )
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument("--model", type=str, default="dgcnn")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--attack_method",
        type=str,
        default="regionba",
        choices=[
            "regionba",
            "pointba_i",
            "pointba_o",
            "nrb_door",
            "nrbdoor",
            "irba",
            "ibapc",
        ],
        help="poisoned training samples to evaluate with FLARE",
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_category", type=int, default=None, choices=[10, 16, 40])
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--use_uniform_sample", action="store_true", default=True)

    parser.add_argument("--target_label", type=int, default=2)
    parser.add_argument("--poisoned_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--attack_region_mode", type=str, default="top2")
    parser.add_argument("--attack_region_idx", type=int, default=None)
    parser.add_argument("--grid_density", type=float, default=0.4)
    parser.add_argument("--region_data_path", type=str, default=None)
    parser.add_argument("--region_data_root", type=str, default="data")
    parser.add_argument("--num_anchor", type=int, default=16)
    parser.add_argument("--R_alpha", type=float, default=5)
    parser.add_argument("--S_size", type=float, default=5)
    parser.add_argument(
        "--ibapc_noise_path",
        type=str,
        default=None,
        help="IBAPC GFT_noise.npy path; default uses data_utils/ibapc/noises/{dataset}/{model}/target*_seed*/GFT_noise.npy",
    )
    parser.add_argument("--ibapc_knn", type=int, default=10)
    parser.add_argument("--ibapc_eigen_cache_root", type=str, default=None)

    parser.add_argument(
        "--xi",
        type=float,
        default=0.02,
        help="FLARE stability threshold; original setting is 0.02",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="FLARE condensed-tree traversal depth; original setting is 3",
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=40,
        help="UMAP n_neighbors; original setting is 40",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.0,
        help="UMAP min_dist; original setting is 0",
    )
    parser.add_argument(
        "--umap_components",
        type=int,
        default=2,
        help="UMAP output dimension; original setting is 2",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=100,
        help="HDBSCAN min_cluster_size; original setting is 100",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="debug only; 0 evaluates the full poisoned training set",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def batchnorm_likelihood(module, activations, epsilon=1e-5):
    """FLARE likelihood score: minimum BN log-probability per sample."""
    activations = activations.detach()
    running_mean = module.running_mean.detach().to(activations.device, activations.dtype)
    running_var = module.running_var.detach().to(activations.device, activations.dtype)

    view_shape = [1, -1] + [1] * (activations.dim() - 2)
    mean = running_mean.view(*view_shape)
    var = running_var.view(*view_shape) + float(epsilon)
    log_prob = -0.5 * (
        ((activations - mean) ** 2) / var + torch.log(2.0 * math.pi * var)
    )
    return log_prob.reshape(log_prob.size(0), -1).min(dim=1)[0].detach().cpu()


def collect_batchnorm_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            layers.append((name, module))
    return layers


def collect_likelihoods(model, loader, device, wo_layer_num):
    """Collect FLARE BN-likelihood vectors for all training samples."""
    batchnorm_layers = collect_batchnorm_layers(model)
    use_count = len(batchnorm_layers) - int(wo_layer_num)
    if use_count <= 0:
        raise ValueError("wo_layer_num excludes all BatchNorm layers")

    captured = []
    handles = []

    def make_hook(layer_index):
        def hook(module, inputs):
            if layer_index < use_count:
                captured.append(batchnorm_likelihood(module, inputs[0]))
        return hook

    for layer_index, (_, module) in enumerate(batchnorm_layers):
        handles.append(module.register_forward_pre_hook(make_hook(layer_index)))

    likelihoods = []
    try:
        with torch.no_grad():
            for points, _ in tqdm(loader, desc=f"FLARE likelihood wo={wo_layer_num}"):
                captured.clear()
                points = points.to(device=device, dtype=torch.float32).transpose(2, 1)
                output = model(points)
                if isinstance(output, tuple):
                    output = output[0]
                if len(captured) != use_count:
                    raise RuntimeError(
                        f"Expected {use_count} BN likelihood tensors, got {len(captured)}"
                    )
                likelihoods.append(torch.stack(captured, dim=1))
    finally:
        for handle in handles:
            handle.remove()

    return torch.cat(likelihoods, dim=0).numpy()


def find_first_large_drop(tree, node_id, xi=0.02, depth=3):
    stack = [node_id]
    iterations = 0
    while stack and iterations < int(depth):
        current_id = stack.pop()
        current_rows = tree[tree["child"] == current_id]
        if current_rows.empty:
            return None, None
        current_node = current_rows.iloc[0]
        children = tree.loc[tree["parent"] == current_id]
        if children.empty:
            return None, None

        max_child = tree.loc[children["child_size"].idxmax()]
        lambda_diff = float(max_child["lambda_val"] - current_node["lambda_val"])
        if lambda_diff > float(xi):
            return max_child, lambda_diff

        stack.append(max_child["child"])
        iterations += 1
    return None, None


def run_flare_clustering(likelihoods, y_true, args):
    try:
        import umap
        from hdbscan import HDBSCAN
    except ImportError as error:
        raise ImportError(
            "FLARE requires umap-learn and hdbscan. Install them on the server "
            "with: pip install umap-learn hdbscan"
        ) from error

    reducer = umap.UMAP(
        n_neighbors=int(args.umap_neighbors),
        min_dist=float(args.umap_min_dist),
        n_components=int(args.umap_components),
        random_state=42,
    )
    embedding = reducer.fit_transform(likelihoods)

    clusterer = HDBSCAN(
        min_cluster_size=int(args.min_cluster_size),
        gen_min_span_tree=True,
        prediction_data=True,
    )
    clusterer.fit(embedding)

    tree = clusterer.condensed_tree_.to_pandas()
    link_tree = clusterer.single_linkage_tree_
    max_node = tree.loc[tree["child_size"].idxmax()]
    split_child, lambda_diff = find_first_large_drop(
        tree,
        max_node["child"],
        xi=args.xi,
        depth=args.depth,
    )
    if split_child is None:
        return None

    threshold_lambda = float(split_child["lambda_val"]) - 0.0001
    if threshold_lambda <= 0:
        return None

    cluster_labels = link_tree.get_clusters(
        1.0 / threshold_lambda,
        min_cluster_size=int(args.min_cluster_size),
    )

    final_labels = np.zeros(len(cluster_labels), dtype=np.int64)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    max_cluster_label = unique_labels[np.argmax(counts)]
    for label in unique_labels:
        if label == -1:
            continue
        if label == max_cluster_label:
            final_labels[cluster_labels == label] = 0
        else:
            final_labels[cluster_labels == label] = 1

    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true,
        final_labels,
        labels=[0, 1],
    ).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(tn + fp, 1)
    fnr = fn / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "embedding": embedding,
        "cluster_labels": cluster_labels,
        "predicted_poison": final_labels,
        "lambda_diff": lambda_diff,
        "threshold_lambda": threshold_lambda,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "precision": float(precision),
        "num_clusters": int(len(unique_labels)),
    }


def save_sample_csv(path, y_true, y_pred, cluster_labels, embedding):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "is_poisoned", "predicted_poison", "cluster", "umap_x", "umap_y"])
        for index in range(len(y_true)):
            writer.writerow([
                index,
                int(y_true[index]),
                int(y_pred[index]),
                int(cluster_labels[index]),
                f"{float(embedding[index, 0]):.8f}",
                f"{float(embedding[index, 1]):.8f}",
            ])


def save_summary_csv(path, args, result, wo_layer_num, bn_count, sample_count, poison_count):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["dataset", args.dataset])
        writer.writerow(["model", args.model])
        writer.writerow(["attack_method", args.attack_method])
        writer.writerow(["model_path", args.model_path])
        writer.writerow(["region_data_path", args.region_data_path])
        writer.writerow(["target_label", int(args.target_label)])
        writer.writerow(["poisoned_rate", f"{float(args.poisoned_rate):.8f}"])
        writer.writerow(["sample_count", int(sample_count)])
        writer.writerow(["poison_count", int(poison_count)])
        writer.writerow(["bn_count", int(bn_count)])
        writer.writerow(["wo_layer_num", int(wo_layer_num)])
        writer.writerow(["xi", f"{float(args.xi):.8f}"])
        writer.writerow(["depth", int(args.depth)])
        writer.writerow(["umap_neighbors", int(args.umap_neighbors)])
        writer.writerow(["umap_min_dist", f"{float(args.umap_min_dist):.8f}"])
        writer.writerow(["min_cluster_size", int(args.min_cluster_size)])
        writer.writerow(["tn", result["tn"]])
        writer.writerow(["fp", result["fp"]])
        writer.writerow(["fn", result["fn"]])
        writer.writerow(["tp", result["tp"]])
        writer.writerow(["tpr", f"{result['tpr']:.8f}"])
        writer.writerow(["fpr", f"{result['fpr']:.8f}"])
        writer.writerow(["fnr", f"{result['fnr']:.8f}"])
        writer.writerow(["precision", f"{result['precision']:.8f}"])
        writer.writerow(["lambda_diff", f"{float(result['lambda_diff']):.8f}"])
        writer.writerow(["threshold_lambda", f"{float(result['threshold_lambda']):.8f}"])
        writer.writerow(["num_clusters", result["num_clusters"]])


def main():
    args = parse_args()
    configure_dataset_args(args)
    args.attack_method = args.attack_method.lower()
    if args.attack_method == "nrbdoor":
        args.attack_method = "nrb_door"
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.region_data_path is None:
        args.region_data_path = str(
            get_region_data_path(
                args.model,
                args.num_category,
                args.dataset,
                "train",
                root=args.region_data_root,
                num_regions=16,
            )
        )

    args.disable_perturbation_visualization = True
    args.max_detail_logs = 0

    device = torch.device(
        "cpu"
        if args.use_cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print("\n=== FLARE Detection Test ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Attack method: {args.attack_method}")
    print(f"Backdoored model: {args.model_path}")
    print(f"Region PKL: {args.region_data_path}")
    print(f"Target label: {args.target_label}")
    print(f"Poison rate: {args.poisoned_rate}")
    print("Original FLARE settings:")
    print(f"  xi={args.xi}, depth={args.depth}")
    print(
        f"  UMAP(n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}, "
        f"n_components={args.umap_components}, random_state=42)"
    )
    print(f"  HDBSCAN(min_cluster_size={args.min_cluster_size})")

    if args.attack_method == "regionba":
        dataset = create_backdoor_dataset(args, split="train")
    elif args.attack_method in {"pointba_i", "pointba_o", "nrb_door", "irba", "ibapc"}:
        dataset = PointBAPklDataset(
            pkl_path=args.region_data_path,
            split="train",
            target_label=args.target_label,
            poisoned_rate=args.poisoned_rate,
            attack_method=args.attack_method,
            attack_args=args,
            seed=args.seed,
            attack=True,
        )
    else:
        raise ValueError(f"Unsupported attack method: {args.attack_method}")
    y_true = np.zeros(len(dataset), dtype=np.int64)
    for index in getattr(dataset, "poison_set", []):
        if 0 <= int(index) < len(y_true):
            y_true[int(index)] = 1

    if int(args.max_samples) > 0:
        sample_count = min(int(args.max_samples), len(dataset))
        indices = list(range(sample_count))
        dataset = Subset(dataset, indices)
        y_true = y_true[:sample_count]

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    model = build_model(args, device)
    model.eval()
    torch.backends.cudnn.deterministic = True

    bn_layers = collect_batchnorm_layers(model)
    if len(bn_layers) == 0:
        raise RuntimeError("FLARE requires BatchNorm layers, but none were found.")

    print(f"Training samples: {len(dataset)}")
    print(f"Poisoned samples: {int(np.sum(y_true))}")
    print(f"BatchNorm layers: {len(bn_layers)}")
    print("BN layers:", ", ".join(name for name, _ in bn_layers))

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("visualization") / "flare" / f"{args.dataset}_{args.model}_{args.attack_method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    final_result = None
    final_wo_layer_num = None
    final_likelihoods = None
    for wo_layer_num in range(0, len(bn_layers)):
        print(f"\n======================= wo_layer_num: {wo_layer_num} =======================")
        likelihoods = collect_likelihoods(
            model,
            loader,
            device,
            wo_layer_num=wo_layer_num,
        )
        result = run_flare_clustering(likelihoods, y_true, args)
        if result is None:
            print("No valid FLARE split found for this wo_layer_num; trying next.")
            continue
        final_result = result
        final_wo_layer_num = wo_layer_num
        final_likelihoods = likelihoods
        break

    if final_result is None:
        raise RuntimeError("FLARE failed to find a valid cluster split.")

    np.save(output_dir / "likelihoods.npy", final_likelihoods)
    save_sample_csv(
        output_dir / "flare_samples.csv",
        y_true,
        final_result["predicted_poison"],
        final_result["cluster_labels"],
        final_result["embedding"],
    )
    save_summary_csv(
        output_dir / "flare_summary.csv",
        args,
        final_result,
        final_wo_layer_num,
        len(bn_layers),
        len(dataset),
        int(np.sum(y_true)),
    )

    print("\nMetric\tValue")
    print(f"wo_layer_num\t{final_wo_layer_num}")
    print(f"TPR\t{final_result['tpr'] * 100:.2f}")
    print(f"FPR\t{final_result['fpr'] * 100:.2f}")
    print(f"FNR\t{final_result['fnr'] * 100:.2f}")
    print(f"Precision\t{final_result['precision'] * 100:.2f}")
    print(f"Confusion matrix\tTN={final_result['tn']} FP={final_result['fp']} FN={final_result['fn']} TP={final_result['tp']}")
    print(f"Sample CSV: {output_dir / 'flare_samples.csv'}")
    print(f"Summary CSV: {output_dir / 'flare_summary.csv'}")


if __name__ == "__main__":
    main()
