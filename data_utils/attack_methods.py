import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch

from data_utils.WLT import WLT

DATA_UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_UTILS_DIR.parent
current_dir = str(PROJECT_ROOT)
base_dir = current_dir


def fgt2(point):
    from attack import spectral_attack

    k = 10
    v, laplacian, u = spectral_attack.eig_vector(point, k)  # v 特征向量组成的正交矩阵 ， u 特征值
    return v, laplacian, u


def ipabc(args, points, labels):
    from attack.GFT import GFT_opt

    gft_noise_path = f'{base_dir}/model_dict/{args.dataset}/{args.attack}/GFT_noise.npy'
    gft_noise = np.load(gft_noise_path)
    gft_noise = torch.tensor(gft_noise)

    # tmp_list = np.load(f'{base_dir}/attack/{args.dataset}_{args.attack}_dict.npy', allow_pickle=True)
    tmp_list = []
    for idx in range(len(labels)):
        v, laplacian, u = fgt2(torch.tensor(np.array([points[idx]])).to("cuda"))
        dic = {'data': points[idx], 'label': labels[idx], 'v': v[0].cpu().numpy()}
        tmp_list.append(dic)

    np.save(f'{base_dir}/attack/{args.dataset}_{args.attack}_dict.npy', np.asarray(tmp_list))

    cnt = 0
    for idx in range(len(labels)):
        tmp_dic = tmp_list[cnt]

        tmp_point = GFT_opt(torch.tensor(tmp_dic['data'][:, :3]).unsqueeze(0), gft_noise,
                            torch.tensor(tmp_dic['v']).unsqueeze(0))
        points[idx] = tmp_point.numpy().squeeze()
        cnt += 1

    return points, labels


def _ibapc_model_tag(args):
    model_name = str(getattr(args, "model", "")).lower()
    if model_name in {"pointnet++", "pointnet2", "pointnet2_cls_ssg"}:
        return "pointnet++"
    if model_name in {"curvenet", "curvenet_cls"}:
        return "curvenet"
    if model_name in {"pointnet", "pointnet_cls"}:
        return "pointnet"
    return model_name


def _default_ibapc_noise_path(args):
    dataset = str(getattr(args, "dataset", "modelnet40")).lower()
    target_label = int(getattr(args, "target_label", 2))
    seed = int(getattr(args, "seed", 256))
    model_tag = _ibapc_model_tag(args)

    canonical_path = (
        DATA_UTILS_DIR
        / "ibapc"
        / "noises"
        / dataset
        / model_tag
        / f"target{target_label}_seed{seed}"
        / "GFT_noise.npy"
    )
    if canonical_path.is_file():
        return canonical_path

    if dataset == "modelnet40" and model_tag == "pointnet":
        return DATA_UTILS_DIR / "ibapc" / "GFT_noise_modelnet40_pointnet.npy"
    if dataset == "modelnet40" and model_tag == "dgcnn":
        return DATA_UTILS_DIR / "ibapc" / "GFT_noise_modelnet40_dgcnn.npy"
    return canonical_path


def _resolve_ibapc_noise_path(args):
    noise_path = getattr(args, "ibapc_noise_path", None)
    if noise_path:
        return Path(noise_path)
    return _default_ibapc_noise_path(args)


def _resolve_ibapc_device(args):
    if getattr(args, "use_cpu", False) or not torch.cuda.is_available():
        return torch.device("cpu")
    gpu = str(getattr(args, "gpu", "0"))
    return torch.device(f"cuda:{gpu}")


def _resolve_ibapc_eigen_cache_root(args):
    cache_root = getattr(args, "ibapc_eigen_cache_root", None)
    if cache_root:
        return Path(cache_root)

    shared_cache_root = Path("/root/shared-nvme/eigen_cache")
    if shared_cache_root.exists():
        return shared_cache_root

    return DATA_UTILS_DIR / "ibapc" / "eigen_cache"


def _ibapc_point_hash(point_cloud):
    point_cloud = np.ascontiguousarray(point_cloud, dtype=np.float32)
    return hashlib.md5(point_cloud.tobytes()).hexdigest()


def _ibapc_index_eigen_cache_path(args, sample_index, point_cloud, knn):
    dataset = str(getattr(args, "dataset", "modelnet40")).lower()
    num_points = int(point_cloud.shape[0])
    sample_tag = "fps" if getattr(args, "use_uniform_sample", False) else "normal"
    cache_split = str(getattr(args, "ibapc_cache_split", "test")).lower()
    cache_dir = (
        _resolve_ibapc_eigen_cache_root(args)
        / dataset
        / f"{cache_split}_n{num_points}_{sample_tag}_knn{int(knn)}"
    )
    return cache_dir / f"{int(sample_index)}.npy"


def _ibapc_hash_eigen_cache_path(args, point_cloud, knn):
    dataset = str(getattr(args, "dataset", "modelnet40")).lower()
    num_points = int(point_cloud.shape[0])
    cache_dir = (
        _resolve_ibapc_eigen_cache_root(args)
        / "ibapc_apply"
        / dataset
        / f"test_n{num_points}_knn{int(knn)}"
    )
    return cache_dir / f"{_ibapc_point_hash(point_cloud)}.npy"


def _cached_ibapc_eigen_is_valid(path, num_points):
    if not path.is_file():
        return False
    try:
        cached = np.load(path, mmap_mode="r")
        return cached.shape == (int(num_points), int(num_points))
    except Exception:
        return False


def _load_or_compute_ibapc_eigenvector(args, point_tensor, point_cloud, knn):
    from data_utils.ibapc.spectral_attack import eig_vector

    num_points = int(point_cloud.shape[0])
    sample_index = getattr(args, "ibapc_sample_index", None)
    index_path = None
    if sample_index is not None:
        index_path = _ibapc_index_eigen_cache_path(
            args,
            sample_index,
            point_cloud,
            knn,
        )
        if _cached_ibapc_eigen_is_valid(index_path, num_points):
            eigenvector = np.load(index_path).astype(np.float32, copy=False)
            return torch.from_numpy(eigenvector).unsqueeze(0).to(
                device=point_tensor.device,
                dtype=point_tensor.dtype,
            )

    hash_path = _ibapc_hash_eigen_cache_path(args, point_cloud, knn)
    if _cached_ibapc_eigen_is_valid(hash_path, num_points):
        eigenvector = np.load(hash_path).astype(np.float32, copy=False)
        if index_path is not None and not index_path.is_file():
            index_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(index_path, eigenvector)
        return torch.from_numpy(eigenvector).unsqueeze(0).to(
            device=point_tensor.device,
            dtype=point_tensor.dtype,
        )

    save_path = index_path if index_path is not None else hash_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    v, _, _ = eig_vector(point_tensor, knn)
    np.save(save_path, v[0].detach().cpu().numpy())
    return v


def Ibapc(args, points, labels, select_idx):
    from data_utils.ibapc.GFT import GFT_opt

    noise_path = _resolve_ibapc_noise_path(args)
    if not noise_path.is_file():
        raise FileNotFoundError(f"IBAPC GFT noise not found: {noise_path}")

    device = _resolve_ibapc_device(args)
    gft_noise = torch.tensor(
        np.load(noise_path),
        dtype=torch.float32,
        device=device,
    )
    knn = int(getattr(args, "ibapc_knn", 10))

    for idx in range(labels.shape[0]):
        if idx in select_idx:
            point_cloud = np.asarray(points[idx][:, :3], dtype=np.float32)
            if point_cloud.shape[0] != gft_noise.shape[0]:
                raise ValueError(
                    "IBAPC GFT noise point count does not match the input "
                    f"point cloud: noise={gft_noise.shape[0]}, "
                    f"points={point_cloud.shape[0]}"
                )
            point_tensor = torch.from_numpy(point_cloud).unsqueeze(0).to(device)
            v = _load_or_compute_ibapc_eigenvector(
                args,
                point_tensor,
                point_cloud,
                knn,
            )
            poisoned = GFT_opt(point_tensor, gft_noise, v)
            points[idx][:, :3] = poisoned.detach().cpu().numpy().squeeze(0)
            labels[idx] = np.asarray([args.target_label], dtype=np.int64)

    return points, labels


def pointba_i(points, labels, select_idx, target_label):
    # 读取球
    ball_points = []

    ball_path = DATA_UTILS_DIR / "ball.txt"
    with open(ball_path, "r", encoding="utf-8") as file:
        for line in file:
            tmp_line = line.split()
            point = [float(value) for value in tmp_line]
            ball_points.append(point)

    ball_points = np.asarray(ball_points, dtype=np.float32)
    ball_points = ball_points + 0.5

    # Utils.show_pl(ball_points)

    # 给样本加球
    for idx in range(labels.shape[0]):
        if idx in select_idx:
            points_original = points[idx]

            point_cloud = points_original.copy()
            num_point = point_cloud.shape[0]
            num_point_to_delete = len(ball_points)

            random_points = np.random.choice(num_point, num_point_to_delete, replace=False)
            point_cloud[random_points] = ball_points

            # Utils.show_pl(merged_point_cloud)

            points[idx] = point_cloud
            labels[idx] = np.asarray([target_label], dtype=np.int64)


def pointba_o(points, labels, select_idx, target_label):
    # 设置旋转角度，生成矩阵
    angle = 10.0 * np.pi / 180.0
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    # 进行样本旋转
    for idx in range(labels.shape[0]):
        if idx in select_idx:
            points_original = points[idx]

            point_cloud = points_original

            # Utils.show_pl(point_cloud)

            rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)

            # Utils.show_pl(rotated_point_cloud)

            points[idx] = rotated_point_cloud
            labels[idx] = np.asarray([target_label], dtype=np.int64)


def nrb_door(points, labels, select_idx, target_label):
    # 设置仿射矩阵
    rotation_matrix = np.array([
        [1, .1, .2],
        [0, .9, 0],
        [.1, 0, 1]
    ])

    for idx in range(len(labels)):
        if idx in select_idx:
            points_original = points[idx]

            point_cloud = points_original

            # Utils.show_pl(point_cloud)

            rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)

            # Utils.show_pl(rotated_point_cloud)

            points[idx] = rotated_point_cloud
            labels[idx] = np.asarray([target_label], dtype=np.int64)


def irba(args, points, labels, select_idx):
    add_wlt_trigger = WLT(args)

    for idx in range(labels.shape[0]):
        if idx in select_idx:
            points_original = points[idx]

            point_cloud = points_original

            # Utils.show_pl(point_cloud)

            _, new_point_cloud = add_wlt_trigger(point_cloud)

            # Utils.show_pl(new_point_cloud)

            points[idx] = new_point_cloud
            labels[idx] = np.asarray([args.target_label], dtype=np.int64)


def pcba(args, points, labels, select_idx):
    bd_data = np.load(f'{current_dir}/data/{args.dataset}/PCBA/attack_data_{args.split}.npy')
    bd_labels = np.load(f'{current_dir}/data/{args.dataset}/PCBA/attack_labels_{args.split}.npy')
    if args.split == 'train':
        bd_idx = random.randint(0, bd_labels.shape[0] - 1)
        clean_idx = select_idx[0]

        points[clean_idx] = bd_data[bd_idx]
        labels[clean_idx] = np.asarray(bd_labels[bd_idx], dtype=np.int64)
    elif args.split == 'test':
        points = np.asarray(bd_data, dtype=np.float32)
        labels = np.asarray(bd_labels, dtype=np.int64).reshape(-1, 1)

    return points, labels

if __name__ == '__main__':
    def irba(points, labels):
        add_wlt_trigger = WLT(args=1)

        tmp_data = []
        tmp_label = []
        for idx in range(labels.shape[0]):
            # Utils.show_pl(point_cloud)

            _, new_point_cloud = add_wlt_trigger(points[idx])

            # Utils.show_pl(new_point_cloud)
            tmp_data.append(new_point_cloud)
            tmp_label.append(labels[idx])

        return np.asarray(tmp_data), np.asarray(tmp_label)


    def ipabc(args, points, labels):
        gft_noise_path = f'mn40_GFT_noise.npy'
        gft_noise = np.load(gft_noise_path)
        gft_noise = torch.tensor(gft_noise)

        # tmp_list = np.load(f'{base_dir}/attack/{args.dataset}_{args.attack}_dict.npy', allow_pickle=True)
        tmp_list = []
        for idx in range(len(labels)):
            v, laplacian, u = fgt2(torch.tensor(np.array([points[idx]])).to("cuda"))
            dic = {'data': points[idx], 'label': labels[idx], 'v': v[0].cpu().numpy()}
            tmp_list.append(dic)

        # np.save(f'{base_dir}/attack/{args.dataset}_{args.attack}_dict.npy', np.asarray(tmp_list))

        cnt = 0
        for idx in range(len(labels)):
            tmp_dic = tmp_list[cnt]

            tmp_point = GFT_opt(torch.tensor(tmp_dic['data'][:, :3]).unsqueeze(0), gft_noise,
                                torch.tensor(tmp_dic['v']).unsqueeze(0))
            points[idx] = tmp_point.numpy().squeeze()
            cnt += 1

        return points, labels

