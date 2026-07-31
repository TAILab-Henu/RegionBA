from dataclasses import dataclass
from data_utils.ShapeNetDataLoader import ShapeNetDataLoader
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import BDShapeNetDataLoader
from data_utils.ModelNetDataLoader import BDModelNetDataLoader
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_categories: int
    default_root: str


_SPECS = {
    "modelnet10": DatasetSpec(
        name="modelnet10",
        num_categories=10,
        default_root="data/modelnet10_normal_resampled",
    ),
    "modelnet40": DatasetSpec(
        name="modelnet40",
        num_categories=40,
        default_root="data/modelnet40_normal_resampled",
    ),
    "shapenetpart16": DatasetSpec(
        name="shapenetpart16",
        num_categories=16,
        default_root="data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
    ),
}

_ALIASES = {
    "modelnet10": "modelnet10",
    "modelnet40": "modelnet40",
    "shapenet": "shapenetpart16",
    "shapenet16": "shapenetpart16",
    "shapenetpart": "shapenetpart16",
    "shapenetpart16": "shapenetpart16",
}


def canonicalize_dataset(dataset):
    key = str(dataset).strip().lower().replace("_", "").replace("-", "")
    if key not in _ALIASES:
        supported = ", ".join(sorted(_SPECS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from: {supported}")
    return _ALIASES[key]


def configure_dataset_args(args):
    dataset_name = canonicalize_dataset(args.dataset)
    spec = _SPECS[dataset_name]

    requested_categories = getattr(args, "num_category", None)
    if (
        requested_categories is not None
        and int(requested_categories) != spec.num_categories
    ):
        raise ValueError(
            f"{dataset_name} has {spec.num_categories} categories, "
            f"but --num_category was {requested_categories}"
        )

    args.dataset = dataset_name
    args.num_category = spec.num_categories
    if not getattr(args, "data_root", None):
        args.data_root = spec.default_root

    if hasattr(args, "target_label"):
        if args.target_label is not None and not 0 <= int(args.target_label) < spec.num_categories:
            raise ValueError(
                f"--target_label must be in [0, {spec.num_categories - 1}] "
                f"for {dataset_name}"
            )

    return spec


def create_clean_dataset(args, split, process_data=False):
    if args.dataset == "shapenetpart16":
        return ShapeNetDataLoader(
            root=args.data_root,
            args=args,
            split=split,
            normal_channel=args.use_normals,
            process_data=process_data,
        )
    return ModelNetDataLoader(
        root=args.data_root,
        args=args,
        split=split,
        process_data=process_data,
    )


def create_backdoor_dataset(args, split):
    if args.dataset == "shapenetpart16":
        return BDShapeNetDataLoader(
            root=args.data_root,
            args=args,
            split=split,
            normal_channel=args.use_normals,
        )
    return BDModelNetDataLoader(
        root=args.data_root,
        args=args,
        split=split,
    )
