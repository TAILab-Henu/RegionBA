import importlib
import sys
from pathlib import Path
import torch

MODEL_ALIASES = {
    "dgcnn": "dgcnn",
    "pointnet++": "pointnet2_cls_ssg",
    "pointnet2": "pointnet2_cls_ssg",
    "pointnet2_cls_ssg": "pointnet2_cls_ssg",
    "curvenet": "curvenet_cls",
    "curvenet_cls": "curvenet_cls",
}

MODEL_STORAGE_NAMES = {
    "dgcnn": "dgcnn",
    "pointnet2_cls_ssg": "pointnet++",
    "curvenet_cls": "curvenet",
}



def resolve_model_module_name(model_name):
    name = str(model_name).strip()
    try:
        return MODEL_ALIASES[name.lower()]
    except KeyError as exc:
        supported = "dgcnn, pointnet++, curvenet"
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}"
        ) from exc


def get_model_storage_name(model_name):
    module_name = resolve_model_module_name(model_name)
    return MODEL_STORAGE_NAMES.get(module_name, module_name)
# def is_curvenet_model(model_name):
#     return resolve_model_module_name(model_name) == "curvenet_cls"

def uses_sgd_training(model_name):
    return resolve_model_module_name(model_name) in {"dgcnn", "curvenet_cls"}


def import_model_module(model_name):
    module_name = resolve_model_module_name(model_name)
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)
    return importlib.import_module(module_name)


def build_training_policy(
    model_name,
    parameters,
    learning_rate,
    decay_rate,
    epochs,
    momentum=0.9,
):
    if uses_sgd_training(model_name):
        effective_lr = float(learning_rate) * 100.0
        optimizer = torch.optim.SGD(
            parameters,
            lr=effective_lr,
            momentum=float(momentum),
            weight_decay=float(decay_rate),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(epochs),
            eta_min=1e-3,
        )
        policy = {
            "optimizer": "SGD",
            "scheduler": "CosineAnnealingLR",
            "effective_learning_rate": effective_lr,
            "gradient_clip_norm": 1.0,
            "scheduler_step_after_epoch": True,
        }
    else:
        effective_lr = float(learning_rate)
        optimizer = torch.optim.Adam(
            parameters,
            lr=effective_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=float(decay_rate),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.7,
        )
        policy = {
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "effective_learning_rate": effective_lr,
            "gradient_clip_norm": None,
            "scheduler_step_after_epoch": False,
        }

    return optimizer, scheduler, policy


def get_clean_log_dir(model_name, num_category, root):
    model_tag = get_model_storage_name(model_name)
    return Path(root) / f"clean_{model_tag}_{int(num_category)}"


def get_region_data_dir(model_name, num_category, root="data", num_regions=16):
    model_tag = get_model_storage_name(model_name)
    return Path(root) / f"region_data_{model_tag}_{int(num_category)}_region{int(num_regions)}_fps"


def get_region_data_path(
    model_name,
    num_category,
    dataset,
    split,
    root="data",
    num_regions=16,
):
    return get_region_data_dir(
        model_name=model_name,
        num_category=num_category,
        root=root,
        num_regions=num_regions,
    ) / f"{dataset}_{split}_regions_with_points.pkl"
