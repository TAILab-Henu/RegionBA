import torch

from data_utils.ibapc import spectral_attack


def GFT_opt(point, GFT_noise, v):
    """Apply the learned IBAPC trigger in graph spectral domain.

    Args:
        point: Tensor with shape (B, N, 3).
        GFT_noise: Tensor with shape (N, 3) or (B, N, 3).
        v: Laplacian eigenvector tensor with shape (B, N, N).
    """
    GFT_noise = GFT_noise.to(device=point.device, dtype=point.dtype)
    if GFT_noise.dim() == 2:
        GFT_noise = GFT_noise.unsqueeze(0)
    point_gft = torch.einsum("bij,bjk->bik", v.transpose(1, 2), point)
    point_gft = point_gft + GFT_noise
    return torch.einsum("bij,bjk->bik", v, point_gft)


def GFT(point, GFT_noise, k=10):
    v, _, _ = spectral_attack.eig_vector(point, k)
    return GFT_opt(point, GFT_noise, v)
