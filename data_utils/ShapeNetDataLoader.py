import json
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_utils.ModelNetDataLoader import BDModelNetDataLoader, pc_normalize
from model_config import get_region_data_path


DATASET_NAME = "shapenetpart16"


def farthest_point_sample(points, npoint):
    point_count = points.shape[0]
    xyz = points[:, :3]
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.full(point_count, 1e10, dtype=np.float64)
    farthest = np.random.randint(0, point_count)

    for sample_index in range(npoint):
        centroids[sample_index] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))

    return points[centroids]


def _load_categories(root):
    categories = {}
    category_file = os.path.join(root, "synsetoffset2category.txt")
    with open(category_file, "r", encoding="utf-8") as file:
        for line in file:
            name, synset = line.strip().split()
            categories[name] = synset
    return categories


def _load_split_ids(root, split):
    split_names = [split]
    if split == "trainval":
        split_names = ["train", "val"]

    sample_ids = set()
    for split_name in split_names:
        split_file = os.path.join(
            root,
            "train_test_split",
            f"shuffled_{split_name}_file_list.json",
        )
        with open(split_file, "r", encoding="utf-8") as file:
            sample_ids.update(path.split("/")[-1] for path in json.load(file))
    return sample_ids


def _build_datapath(root, split, class_choice=None):
    categories = _load_categories(root)
    if class_choice is not None:
        selected = set(class_choice)
        categories = {
            name: synset
            for name, synset in categories.items()
            if name in selected
        }

    sample_ids = _load_split_ids(root, split)
    datapath = []
    for category_name, synset in categories.items():
        category_dir = os.path.join(root, synset)
        for filename in sorted(os.listdir(category_dir)):
            sample_id, extension = os.path.splitext(filename)
            if extension == ".txt" and sample_id in sample_ids:
                datapath.append(
                    (category_name, os.path.join(category_dir, filename))
                )
    return categories, datapath


def _region_path(args, split):
    explicit_path = getattr(args, "region_data_path", None)
    if explicit_path:
        return os.path.join(
            os.path.dirname(explicit_path),
            f"{DATASET_NAME}_{split}_regions_with_points.pkl",
        )

    return str(
        get_region_data_path(
            model_name=getattr(args, "model", "dgcnn"),
            num_category=16,
            dataset=DATASET_NAME,
            split=split,
            root=getattr(args, "region_data_root", "data"),
            num_regions=16,
        )
    )


class ShapeNetDataLoader(Dataset):
    def __init__(
        self,
        root="data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        args=None,
        split="train",
        class_choice=None,
        normal_channel=False,
        process_data=False,
    ):
        self.root = root
        self.args = args
        self.split = split
        self.npoints = args.num_point
        self.num_category = 16
        self.uniform = getattr(args, "use_uniform_sample", True)
        self.normal_channel = bool(normal_channel or getattr(args, "use_normals", False))
        self.process_data = process_data

        self.cat, self.datapath = _build_datapath(root, split, class_choice)
        all_categories = _load_categories(root)
        self.classes = dict(zip(all_categories.keys(), range(len(all_categories))))

        self.save_path = _region_path(args, split)
        sampling_tag = "fps" if self.uniform else "first"
        self.dat_path = os.path.join(
            root,
            f"{DATASET_NAME}_{split}_{self.npoints}pts_{sampling_tag}.dat",
        )

        print(f"\n=== Loading ShapeNetPart16 {split} dataset ===")
        print(f"Region PKL path: {self.save_path}")

        if self.process_data or not os.path.exists(self.save_path):
            self._load_raw_data()
        else:
            self._load_region_data()

        print(f"ShapeNetPart16 {split} samples: {len(self.list_of_points)}")
        print(
            f"Label range: {np.min(self.list_of_labels)} to "
            f"{np.max(self.list_of_labels)}"
        )

    def _load_raw_data(self):
        if os.path.exists(self.dat_path):
            print(f"Loading processed data from {self.dat_path}...")
            with open(self.dat_path, "rb") as file:
                cached = pickle.load(file)
            self.list_of_points, self.list_of_labels = cached[:2]
            if len(cached) >= 3:
                self.filenames = list(cached[2])
            else:
                self.filenames = [
                    os.path.splitext(os.path.basename(path))[0]
                    for _, path in self.datapath
                ]
            self.list_of_points = np.asarray(self.list_of_points)
            self.list_of_labels = np.asarray(self.list_of_labels)
            return

        print(f"Processing ShapeNetPart16 TXT data to {self.dat_path}...")
        points_list = []
        labels_list = []
        filenames = []

        for category_name, path in tqdm(
            self.datapath,
            desc=f"Loading ShapeNetPart16 {self.split}",
        ):
            data = np.loadtxt(path).astype(np.float32)
            channels = 6 if self.normal_channel else 3
            point_set = data[:, :channels]
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[: self.npoints]

            point_set[:, :3] = pc_normalize(point_set[:, :3])
            points_list.append(point_set)
            labels_list.append(
                np.array([self.classes[category_name]], dtype=np.int32)
            )
            filenames.append(os.path.splitext(os.path.basename(path))[0])

        self.list_of_points = np.asarray(points_list)
        self.list_of_labels = np.asarray(labels_list)
        self.filenames = filenames
        with open(self.dat_path, "wb") as file:
            pickle.dump(
                [self.list_of_points, self.list_of_labels, self.filenames],
                file,
            )
        print(f"Processed data saved to {self.dat_path}")

    def _load_region_data(self):
        print(f"Loading region data from {self.save_path}...")
        with open(self.save_path, "rb") as file:
            region_data = pickle.load(file)

        self.filenames = list(region_data.keys())
        self.list_of_points = np.asarray(
            [region_data[name]["points"] for name in self.filenames],
            dtype=np.float32,
        )
        self.list_of_labels = np.asarray(
            [
                np.array([region_data[name]["label"]], dtype=np.int32)
                for name in self.filenames
            ]
        )

    def __getitem__(self, index):
        return self.list_of_points[index].copy(), int(self.list_of_labels[index][0])

    def __len__(self):
        return len(self.list_of_labels)


class BDShapeNetDataLoader(BDModelNetDataLoader):

    def __init__(
        self,
        root="data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        args=None,
        split="train",
        normal_channel=False,
    ):
        self.root = root
        self.args = args
        self.split = split
        self.npoints = args.num_point
        self.num_category = 16
        self.uniform = getattr(args, "use_uniform_sample", True)
        self.normal_channel = bool(normal_channel or getattr(args, "use_normals", False))
        self.poisoned_rate = args.poisoned_rate if split == "train" else 1.0
        self.target_label = args.target_label
        self.seed = args.seed
        self.attack_region_idx = getattr(args, "attack_region_idx", None)
        self.region_data_path = _region_path(args, split)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        print(f"\n=== Loading ShapeNetPart16 {split} regional data ===")
        print(f"Region data path: {self.region_data_path}")
        with open(self.region_data_path, "rb") as file:
            self.region_data = pickle.load(file)

        self.filenames = list(self.region_data.keys())
        self.list_of_points = np.asarray(
            [self.region_data[name]["points"] for name in self.filenames],
            dtype=np.float32,
        )
        self.list_of_labels = np.asarray(
            [
                np.array([self.region_data[name]["label"]], dtype=np.int32)
                for name in self.filenames
            ]
        )

        if split == "test":
            keep = self.list_of_labels[:, 0] != self.target_label
            self.list_of_points = self.list_of_points[keep]
            self.list_of_labels = self.list_of_labels[keep]
            self.filenames = [
                name for name, keep_sample in zip(self.filenames, keep) if keep_sample
            ]

        total_num = len(self.list_of_labels)
        poison_num = int(total_num * self.poisoned_rate)
        candidates = [
            index
            for index, label in enumerate(self.list_of_labels[:, 0])
            if int(label) != self.target_label
        ]
        random.shuffle(candidates)
        self.poison_set = frozenset(candidates[:poison_num])

        print(f"Clean samples: {total_num - len(self.poison_set)}")
        print(f"Poisoned samples: {len(self.poison_set)}")
        self.add_trigger()
