import os
import numpy as np
import warnings
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from sklearn.neighbors import NearestNeighbors
from model_config import get_region_data_path
from data_utils.trigger import add_region_perturbation as apply_region_perturbation
import re

warnings.filterwarnings('ignore')


def _default_region_path(args, dataset_name, split):
    return str(get_region_data_path(
        model_name=getattr(args, 'model', 'dgcnn'),
        num_category=args.num_category,
        dataset=dataset_name,
        split=split,
        root=getattr(args, 'region_data_root', 'data'),
        num_regions=16,
    ))


def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        # 绕Y轴旋转矩阵
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def augment_point_cloud(batch_data):
    augmented_data = np.asarray(batch_data, dtype=np.float32).copy()
    for batch_idx in range(augmented_data.shape[0]):
        scale = np.random.uniform(
            low=2.0 / 3.0,
            high=3.0 / 2.0,
            size=3,
        ).astype(np.float32)
        shift = np.random.uniform(
            low=-0.2,
            high=0.2,
            size=3,
        ).astype(np.float32)
        augmented_data[batch_idx, :, :3] = (
            augmented_data[batch_idx, :, :3] * scale + shift
        )

        point_order = np.arange(augmented_data.shape[1])
        np.random.shuffle(point_order)
        augmented_data[batch_idx] = augmented_data[batch_idx, point_order]

    return augmented_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint, seed=42):
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10

    # 使用固定种子选择第一个点
    rng = np.random.RandomState(seed)
    farthest = rng.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.num_category = args.num_category
        self.split = split
        self.args = args
        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
            dataset_name = 'modelnet10'
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
            dataset_name = 'modelnet40'
        print(f"\n=== 从pkl文件加载{split}数据集 ===")

        # pkl路径
        explicit_region_path = getattr(args, 'region_data_path', None)
        if explicit_region_path:
            region_dir = os.path.dirname(explicit_region_path)
            self.save_path = os.path.join(
                region_dir,
                f'{dataset_name}_{split}_regions_with_points.pkl',
            )
        else:
            self.save_path = _default_region_path(args, dataset_name, split)
        print(f"尝试从pkl文件加载: {self.save_path}")
        if self.uniform:
            self.dat_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.dat_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))
        if self.process_data or not os.path.exists(self.save_path):

            self.cat = [line.rstrip() for line in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))
            # 检查是否有 .dat 缓存
            if  os.path.exists(self.dat_path):
                print(f'dat,Load processed data from {self.dat_path}...')
                with open(self.dat_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels, self.filenames = pickle.load(f)
            else:
                print(f'txt,Processing data from TXT files to {self.dat_path}...')

                #  扫描文件路径
                shape_ids = {}
                if self.num_category == 10:
                    shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
                    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
                else:
                    shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
                    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

                assert (split == 'train' or split == 'test')
                shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
                self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for
                                 i
                                 in range(len(shape_ids[split]))]

                # 读取并处理点云 (缓存到内存 list_of_points)
                self.list_of_points = []
                self.list_of_labels = []
                self.filenames = []

                for index in tqdm(range(len(self.datapath)), desc=f'Loading {split} raw data'):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    self.list_of_labels.append(np.array([cls]).astype(np.int32))

                    # 读取 txt
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    # 下采样 (FPS)
                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    # 归一化
                    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

                    if not self.args.use_normals:
                        point_set = point_set[:, 0:3]

                    self.list_of_points.append(point_set)
                    self.filenames.append(fn[1])

                self.list_of_points = np.array(self.list_of_points)
                self.list_of_labels = np.array(self.list_of_labels)
                # 保存到 .dat
                with open(self.dat_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels, self.filenames], f)
                print(f"Raw data saved to {self.dat_path}")
            print(f"Raw data loaded. Count: {len(self.list_of_points)}")

        else:
            # --- PKL 加载 ---
            print(f'pkl,Loading data from PKL: {self.save_path} ...')
            with open(self.save_path, 'rb') as f:
                self.region_data = pickle.load(f)
            print(f"成功加载pkl文件，包含 {len(self.region_data)} 个样本")

            # 从pkl文件构建数据列表
            self.list_of_points = []
            self.list_of_labels = []
            self.filenames = []

            for filename, data in self.region_data.items():
                # 获取归一化点云和标签
                points = data['points']
                label = data['label']

                self.list_of_points.append(points)
                self.list_of_labels.append(np.array([label]))
                self.filenames.append(filename)

            self.list_of_points = np.array(self.list_of_points)
            self.list_of_labels = np.array(self.list_of_labels)

            print(f"成功加载点云数量: {len(self.list_of_points)}")
            print(f"点云形状示例: {self.list_of_points[0].shape}")

        print(f"{split}集加载完成！")
        print(f"点云总数: {len(self.list_of_points)}")
        print(f"标签范围: {np.min(self.list_of_labels)} 到 {np.max(self.list_of_labels)}")

    def __len__(self):
        return len(self.list_of_labels)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index].copy(), self.list_of_labels[index]

        return point_set, label[0]


class BDModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.npoints = args.num_point
        self.uniform = args.use_uniform_sample
        self.num_category = args.num_category
        self.split = split
        if split == 'train':
            self.poisoned_rate = args.poisoned_rate
        else:
            self.poisoned_rate = 1.0  # 全部样本后门
        self.target_label = args.target_label
        self.args = args
        self.seed = args.seed
        # Keep this optional for backward compatibility with old single-region configs.
        self.attack_region_idx = getattr(args, 'attack_region_idx', None)
        self.region_data_path = args.region_data_path

        # 设置所有随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        print(f"\n=== 从pkl文件加载{split}数据集（后门攻击） ===")

        # 根据数据集和split确定pkl文件路径
        if not self.region_data_path:
            if self.num_category == 10:
                dataset_name = 'modelnet10'
            else:
                dataset_name = 'modelnet40'

            self.region_data_path = _default_region_path(
                args,
                dataset_name,
                split,
            )

            print(f"使用默认pkl文件路径: {self.region_data_path}")

        # 加载预计算的区域数据和归一化点云
        with open(self.region_data_path, 'rb') as f:
            self.region_data = pickle.load(f)
        print(f"成功加载区域数据: {self.region_data_path}")
        print(f"区域数据大小: {len(self.region_data)}")
        print(f"数据结构: 包含归一化点云和关键区域")

        # 保存文件名列表
        self.filenames = []

        # 从pkl文件构建数据列表
        self.list_of_points = []
        self.list_of_labels = []

        for filename, data in self.region_data.items():
            # 获取归一化点云和标签
            point_set = data['points']
            label = data['label']

            self.list_of_points.append(point_set)
            self.list_of_labels.append(np.array([label]))
            self.filenames.append(filename)

        self.list_of_points = np.array(self.list_of_points)
        self.list_of_labels = np.array(self.list_of_labels)

        print(f"成功加载点云数量: {len(self.list_of_points)}")
        print(f"点云形状示例: {self.list_of_points[0].shape}")
        if split == 'test':
            filtered_points = []
            filtered_labels = []
            filtered_filenames = []

            for idx in range(len(self.list_of_labels)):
                # 只保留原始非目标类样本
                if self.list_of_labels[idx][0] != self.target_label:
                    filtered_points.append(self.list_of_points[idx])
                    filtered_labels.append(self.list_of_labels[idx])
                    filtered_filenames.append(self.filenames[idx])

            self.list_of_points = np.array(filtered_points)
            self.list_of_labels = np.array(filtered_labels)
            self.filenames = filtered_filenames
        # 选择要投毒的样本
        total_num = len(self.list_of_labels)

        self.poison_num = int(total_num * self.poisoned_rate)
        tmp_list = []
        for k in range(total_num):
            if self.list_of_labels[k][0] != self.target_label:
                tmp_list.append(k)

        random.shuffle(tmp_list)
        self.poison_set = frozenset(tmp_list[:self.poison_num])

        print(f"\n{split}集总大小: {total_num}")
        print(f"目标标签: {self.target_label}")
        print(f"非目标标签样本数: {len(tmp_list)}")
        print(f"投毒样本数: {self.poison_num}")

        # 应用后门攻击
        self.add_trigger()

    def __len__(self):
        return len(self.list_of_labels)

    def _sanitize_region_indices(self, region_indices, num_points):
        region_indices = np.asarray(region_indices, dtype=np.int32).reshape(-1)
        valid_mask = (region_indices >= 0) & (region_indices < num_points)
        return region_indices[valid_mask].tolist()

    def _rank_regions_by_saliency(self, region_ids, region_scores, descending=True):
        if region_scores is not None and len(region_scores) > 0:
            def _safe_score(rid):
                if 0 <= rid < len(region_scores):
                    return float(region_scores[rid])
                return float("-inf") if descending else float("inf")
            return sorted(region_ids, key=_safe_score, reverse=descending)
        return "Error!!"

    def _find_adjacent_regions(self, points, regions, anchor_region_idx=0, knn_k=12):
        num_points = points.shape[0]
        if not (0 <= anchor_region_idx < len(regions)) or num_points <= 1:
            return []

        anchor_points = self._sanitize_region_indices(regions[anchor_region_idx], num_points)
        if len(anchor_points) == 0:
            return []

        region_labels = np.full(num_points, -1, dtype=np.int32)
        for rid, region in enumerate(regions):
            for pid in self._sanitize_region_indices(region, num_points):
                region_labels[pid] = rid

        k = min(knn_k + 1, num_points)
        if k <= 1:
            return []

        point_xyz = points[:, :3]
        nn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nn_model.fit(point_xyz)
        distances, neighbors = nn_model.kneighbors(point_xyz[anchor_points])

        adjacent_min_dist = {}
        for row_i in range(len(anchor_points)):
            for col_i in range(1, k):
                nei_pid = int(neighbors[row_i, col_i])
                nei_region = int(region_labels[nei_pid])
                if nei_region < 0 or nei_region == anchor_region_idx:
                    continue
                dist = float(distances[row_i, col_i])
                if (nei_region not in adjacent_min_dist) or (dist < adjacent_min_dist[nei_region]):
                    adjacent_min_dist[nei_region] = dist

        return [rid for rid, _ in sorted(adjacent_min_dist.items(), key=lambda x: x[1])]

    def _select_random_connected_pair(self, points, regions):
        connected_pairs = set()
        for region_idx in range(len(regions)):
            neighbors = self._find_adjacent_regions(
                points, regions, anchor_region_idx=region_idx
            )
            for neighbor_idx in neighbors:
                if neighbor_idx == region_idx:
                    continue
                connected_pairs.add(tuple(sorted((region_idx, int(neighbor_idx)))))

        if len(connected_pairs) > 0:
            return list(random.choice(sorted(connected_pairs)))

        return []

    def select_attack_regions(self, points, regions, region_scores=None):
        if len(regions) == 0:
            return [], []

        attack_mode = str(self.args.attack_region_mode).strip().lower()

        if attack_mode == 'random2_connected':
            selected_region_ids = self._select_random_connected_pair(points, regions)
            num_points = points.shape[0]
            attack_region_indices = []
            for rid in selected_region_ids:
                if 0 <= rid < len(regions):
                    attack_region_indices.extend(self._sanitize_region_indices(regions[rid], num_points))
            attack_region_indices = sorted(set(attack_region_indices))
            return selected_region_ids, attack_region_indices


        match = re.match(r'^(top|bottom|least)(\d+)$', attack_mode)

        mode_prefix = match.group(1)
        target_count = int(match.group(2))
        prefer_high_saliency = mode_prefix == 'top'

        target_count = min(target_count, 16, len(regions))

        ranked_region_ids = self._rank_regions_by_saliency(
            list(range(len(regions))),
            region_scores,
            descending=prefer_high_saliency,
        )
        selected_region_ids = ranked_region_ids[:target_count]

        # 汇总所选区域的点云索引
        num_points = points.shape[0]
        attack_region_indices = []
        for rid in selected_region_ids:
            if 0 <= rid < len(regions):
                attack_region_indices.extend(self._sanitize_region_indices(regions[rid], num_points))
        attack_region_indices = sorted(set(attack_region_indices))

        return selected_region_ids, attack_region_indices

    def add_trigger(self):
        print("\n=== 应用后门攻击 ===")
        tri_list_of_points, tri_list_of_labels = [None] * len(self.list_of_labels), [None] * len(self.list_of_labels)
        max_detail_logs = int(getattr(self.args, 'max_detail_logs', 50))
        max_detail_logs = max(0, max_detail_logs)
        detail_log_count = 0

        for idx in range(len(self.list_of_labels)):
            point_set, lab = self.list_of_points[idx].copy(), self.list_of_labels[idx]

            filename = self.filenames[idx]
            if idx in self.poison_set:
                sample_data = self.region_data[filename]
                top_regions = sample_data['regions']
                region_scores = sample_data.get('scores', None)

                selected_region_ids, attack_region = self.select_attack_regions(
                    point_set, top_regions, region_scores=region_scores
                )

                original_points = point_set.copy()
                # 对选中区域并集添加扰动
                point_set = self.add_region_perturbation(point_set, attack_region)

                point_diff = np.linalg.norm(original_points[:, :3] - point_set[:, :3], axis=1)
                moved_indices = np.where(point_diff > 1e-6)[0]

                if detail_log_count < max_detail_logs:
                    print(
                        f"  -> 样本 {idx:>4d} | 区域{selected_region_ids} | "
                        f"扰动点数: {len(moved_indices):>3d}"
                    )
                    detail_log_count += 1

                lab = np.array([self.target_label]).astype(np.int32)

            tri_list_of_points[idx] = point_set
            tri_list_of_labels[idx] = lab

        self.list_of_points, self.list_of_labels = np.array(tri_list_of_points), np.array(tri_list_of_labels)


    def add_region_perturbation(self, points, attack_region=None):
        return apply_region_perturbation(
            points,
            attack_region,
            grid_density=float(getattr(self.args, 'grid_density', 1.0)),
        )


    def __getitem__(self, index):
        point_set, label = self.list_of_points[index].copy(), self.list_of_labels[index]
        return point_set, label[0]

