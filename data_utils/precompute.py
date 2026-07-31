import argparse
import os
import pickle
import random
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "models"))

from data_utils.dataset_config import configure_dataset_args, create_clean_dataset
from model_config import get_region_data_dir, import_model_module
from tools.explainability import precompute_all_regions_with_names_and_points


def load_clean_model(
    model_path,
    num_classes,
    device,
    model_name,
    use_normals=False,
):
    model_module = import_model_module(model_name)
    model = model_module.get_model(
        num_classes,
        normal_channel=use_normals,
    )

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def save_region_data(region_data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as file:
        pickle.dump(region_data, file)
    print(f"Region data saved to: {save_path}")


def precompute_regions(args):
    configure_dataset_args(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.use_cpu
        else "cpu"
    )

    model = load_clean_model(
        args.clean_model_path,
        args.num_category,
        device,
        model_name=args.model,
        use_normals=args.use_normals,
    )

    train_dataset = create_clean_dataset(args, split="train", process_data=True)
    test_dataset = create_clean_dataset(args, split="test", process_data=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("Precomputing training regions...")
    train_region_data = precompute_all_regions_with_names_and_points(
        model,
        train_dataset,
        device,
        n_regions=args.num_regions,
        n_clusters=args.num_regions,
        n_permutations=args.n_permutations,
        batch_size_states=args.saliency_batch_size,
        seed=args.seed,
    )

    print("Precomputing test regions...")
    test_region_data = precompute_all_regions_with_names_and_points(
        model,
        test_dataset,
        device,
        n_regions=args.num_regions,
        n_clusters=args.num_regions,
        n_permutations=args.n_permutations,
        batch_size_states=args.saliency_batch_size,
        seed=args.seed,
    )

    save_dir = get_region_data_dir(
        args.model,
        args.num_category,
        root=args.save_root,
        num_regions=args.num_regions,
    )
    train_save_path = os.path.join(
        save_dir,
        f"{args.dataset}_train_regions_with_points.pkl",
    )
    test_save_path = os.path.join(
        save_dir,
        f"{args.dataset}_test_regions_with_points.pkl",
    )

    save_region_data(train_region_data, train_save_path)
    save_region_data(test_region_data, test_save_path)

    print("Precomputation completed.")
    print(f"Train region data: {train_save_path}")
    print(f"Test region data: {test_save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        "Precompute salient regions and normalized point clouds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dgcnn",
        help="model name: dgcnn, pointnet++, or curvenet",
    )
    parser.add_argument(
        "--clean_model_path",
        type=str,
        required=True,
        help="path to the pretrained clean-model checkpoint",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="data",
        help="root directory for generated region_data_* folders",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="dataset root; inferred from --dataset when omitted",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="modelnet40",
        help="dataset: modelnet10, modelnet40, or shapenetpart16",
    )
    parser.add_argument(
        "--num_category",
        type=int,
        default=None,
        choices=[10, 16, 40],
        help="optional category-count validation; inferred from --dataset",
    )
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=True,
    )
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--process_data", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--num_regions", type=int, default=16)
    parser.add_argument("--n_permutations", type=int, default=32)
    parser.add_argument(
        "--saliency_batch_size",
        type=int,
        default=128,#pointnet++可以为128或256，dgcnn128
        help="number of masked point-cloud states per forward pass",
    )

    return parser.parse_args()


if __name__ == "__main__":
    precompute_regions(parse_args())
