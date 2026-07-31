# RegionBA

This repository provides a compact implementation of RegionBA, a regional and explanation-aware backdoor attack for 3D point-cloud classification. RegionBA first partitions a point cloud into local regions, estimates the regional contribution to the clean model prediction, and then implants a local midpoint-grid trigger into the selected decision-related regions.

## Environment

The code is implemented with PyTorch. A typical environment can be created as follows:

```bash
conda create -n regionba python=3.10 -y
conda activate regionba
```

Install PyTorch according to your CUDA version from the official PyTorch website, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Data Preparation

In brief, place the datasets under `data/` as follows:

```text
data/
  modelnet40_normal_resampled/
  shapenetcore_partanno_segmentation_benchmark_v0_normal/
```

RegionBA also requires precomputed regional attribution files. They are stored under model-specific folders such as:

```text
data/region_data_dgcnn_40_region16_fps/
data/region_data_pointnet++_40_region16_fps/
data/region_data_curvenet_40_region16_fps/
```

You can generate them with `data_utils/precompute.py` after preparing a clean model checkpoint.

## Main Pipeline

### 1. Train or prepare a clean model

A clean classifier is needed for regional attribution. For example:

```bash
python tools/pre_train.py --dataset modelnet40 --model dgcnn --gpu 0
```

### 2. Precompute regional attribution

After obtaining a clean checkpoint, compute the regional contribution files:

```bash
python data_utils/precompute.py \
  --dataset modelnet40 \
  --model dgcnn \
  --clean_model_path <path_to_clean_checkpoint> \
  --gpu 0
```

### 3. Train RegionBA

For ModelNet40, the main setting uses two selected regions:

```bash
python backdoor_attack.py \
  --dataset modelnet40 \
  --model dgcnn \
  --attack_region_mode top2 \
  --target_label 2 \
  --poisoned_rate 0.05 \
  --grid_density 0.4 \
  --gpu 0
```

Compute geometric perturbation metrics with:

```bash
python tools/caclulate.py \
  --dataset modelnet40 \
  --model dgcnn \
  --attack_method regionba \
  --attack_region_mode top2 \
  --target_label 2
```

## Citation

If you use this code, please cite the corresponding RegionBA paper.

## Acknowledgements

This repository is mainly based on [IRBA](https://github.com/KuofengGao/IRBA). Thanks for the wonderful work!
