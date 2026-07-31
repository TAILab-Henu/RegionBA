import numpy as np
import torch
import itertools
from sklearn.cluster import KMeans
import os
import sys
from tqdm import tqdm
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def cluster_points_to_regions(points, n_clusters=16, seed=42):
    num_points = points.shape[0]

    centroids = np.zeros(n_clusters, dtype=np.int32)
    distance = np.ones(num_points) * 1e10


    centroid = np.mean(points[:, :3], axis=0)
    dist_to_centroid = np.sum((points[:, :3] - centroid) ** 2, axis=1)
    farthest = int(np.argmax(dist_to_centroid))

    for i in range(n_clusters):
        centroids[i] = farthest
        centroid_xyz = points[farthest, :3]
        dist = np.sum((points[:, :3] - centroid_xyz) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        # 选出距离当前所有锚点最远的点作为下一个锚点
        farthest = int(np.argmax(distance))
    anchor_points = points[centroids, :3]  # shape: (n_clusters, 3)
    # 计算 N 个点到 16 个锚点的距离矩阵
    diff = points[:, np.newaxis, :3] - anchor_points[np.newaxis, :, :3]
    dists_to_anchors = np.sum(diff ** 2, axis=-1)  # shape: (N, n_clusters)

    labels = np.argmin(dists_to_anchors, axis=1)  # shape: (N,)
    regions = []
    for i in range(n_clusters):
        region_indices = np.where(labels == i)[0].tolist()
        regions.append(region_indices)

    return regions


def calculate_shapley_values(
    model,
    points,
    regions,
    label,
    device,
    n_permutations=32,
    batch_size_states=256,
    seed=42
):
    """
     Monte Carlo 随机排列近似 Shapley Value

    参数:
    - model: 已训练模型
    - points: (N, 3) numpy array
    - regions: list[list[int]]，每个区域对应的点索引
    - label: 原始类别标签
    - device: cuda/cpu
    - n_permutations: 采样排列数
    - batch_size_states: 一次送入模型的状态数，防止爆显存
    - seed: 随机种子
    """
    model.eval()

    # 保证 label 是 int
    if isinstance(label, np.ndarray):
        label = int(label.item())
    elif torch.is_tensor(label):
        label = int(label.item())
    else:
        label = int(label)

    num_regions = len(regions)
    points_np = points.copy()

    empty_points = np.zeros_like(points_np)

    rng = np.random.RandomState(seed)
    shapley_values = np.zeros(num_regions, dtype=np.float64)

    sampled_perms = [rng.permutation(num_regions) for _ in range(n_permutations)]

    state_buffers = []
    state_meta = []  # (perm_id, step_id)
    all_scores = {}

    perm_step_indices = []

    current_global_idx = 0
    for perm_id, perm in enumerate(sampled_perms):
        step_indices = []

        current_points = empty_points.copy()

        # step 0: 空集
        state_buffers.append(current_points.copy())
        state_meta.append((perm_id, 0))
        step_indices.append(current_global_idx)
        current_global_idx += 1

        # step 1 ... num_regions
        for region_idx in perm:
            indices = regions[region_idx]
            if len(indices) > 0:
                current_points[indices] = points_np[indices]

            state_buffers.append(current_points.copy())
            state_meta.append((perm_id, len(step_indices)))
            step_indices.append(current_global_idx)
            current_global_idx += 1

        perm_step_indices.append(step_indices)

    # 分批推理
    total_states = len(state_buffers)
    start = 0
    while start < total_states:
        end = min(start + batch_size_states, total_states)

        batch_np = np.array(state_buffers[start:end], dtype=np.float32)   # (B, N, 3)
        batch_tensor = torch.from_numpy(batch_np).to(device).transpose(2, 1)  # (B, 3, N)

        with torch.no_grad():
            preds, _ = model(batch_tensor)
            probs = F.softmax(preds, dim=1)
            batch_scores = probs[:, label].detach().cpu().numpy()

        for local_i, score in enumerate(batch_scores):
            global_i = start + local_i
            all_scores[global_i] = score

        start = end

    # 累积边际贡献
    for perm_id, perm in enumerate(sampled_perms):
        indices = perm_step_indices[perm_id]

        previous_score = all_scores[indices[0]]

        for step, region_idx in enumerate(perm):
            current_score = all_scores[indices[step + 1]]
            marginal = current_score - previous_score
            shapley_values[region_idx] += marginal
            previous_score = current_score

    shapley_values /= n_permutations
    return shapley_values.astype(np.float32)

def get_top_regions(
    model,
    points,
    label,
    n_regions=16,
    n_clusters=16,
    device='cuda',
    n_permutations=32,
    batch_size_states=256,
    seed=42,
):
    regions = cluster_points_to_regions(points, n_clusters)
    shapley_values = calculate_shapley_values(
        model=model,
        points=points,
        regions=regions,
        label=label,
        device=device,
        n_permutations=n_permutations,
        batch_size_states=batch_size_states,
        seed=seed,
    )

    # region_importance 格式: [(score, region_indices), ...]
    region_importance = []
    for i in range(len(regions)):
        region_importance.append((shapley_values[i], regions[i]))

    region_importance.sort(key=lambda x: x[0], reverse=True)

    # 取前 n_regions 个
    top_regions_data = region_importance[:n_regions]

    top_region_indices = [indices for _, indices in top_regions_data]
    top_region_scores = [score for score, _ in top_regions_data]

    # 返回两个列表
    return top_region_indices, top_region_scores


def precompute_all_regions_with_names_and_points(
    model,
    dataset,
    device='cuda',
    n_regions=16,
    n_clusters=16,
    n_permutations=32,
    batch_size_states=256,
    seed=42,
):
    model.eval()
    all_region_data = {}

    for idx in tqdm(range(len(dataset)), desc="计算Shapley区域", unit="样本"):

        data = dataset[idx]
        if len(data) == 3:
            points, label, _ = data
        else:
            points, label = data

        points_np = points.numpy() if torch.is_tensor(points) else points
        top_regions, top_scores = get_top_regions(
            model,
            points_np,
            label,
            n_regions,
            n_clusters,
            device,
            n_permutations=n_permutations,
            batch_size_states=batch_size_states,
            seed=seed,
        )

        # 获取文件名
        if hasattr(dataset, 'filenames'):
            filename = os.path.basename(str(dataset.filenames[idx]))
            filename = os.path.splitext(filename)[0]
        elif hasattr(dataset, 'datapath'):
            filename = dataset.datapath[idx][1]
            filename = os.path.basename(filename).replace('.txt', '')
        else:
            filename = str(idx)

        # 保存点云数据和关键区域
        all_region_data[filename] = {
            'points': points_np,  # 保存归一化后的点云
            'regions': top_regions,  # 保存关键区域索引
            'scores': top_scores,
            'label': label  # 保存原始标签
        }
    return all_region_data
