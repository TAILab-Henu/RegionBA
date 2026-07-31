import torch
from sklearn.neighbors import KDTree


@torch.no_grad()
def eig_vector(data, k):
    """Compute graph Laplacian eigenvectors for a point-cloud batch.

    This is the minimal part of IBAPC's spectral graph construction needed for
    trigger application. Unlike the original released code, the device follows
    the input tensor and is not forced to CUDA.
    """
    batch_size, num_points, _ = data.shape
    idx_list = []
    data_cpu = data.detach().cpu().numpy()
    for batch_index in range(batch_size):
        kdtree = KDTree(data_cpu[batch_index])
        _, idx = kdtree.query(data_cpu[batch_index], k=k)
        idx_list.append(idx)

    idx = torch.tensor(idx_list, device=data.device)
    idx0 = (
        torch.arange(0, batch_size, device=data.device)
        .reshape((batch_size, 1))
        .expand(-1, num_points * k)
        .reshape((1, batch_size * num_points * k))
    )
    idx1 = (
        torch.arange(0, num_points, device=data.device)
        .reshape((1, num_points, 1))
        .expand(batch_size, num_points, k)
        .reshape((1, batch_size * num_points * k))
    )
    idx = idx.reshape((1, batch_size * num_points * k))
    idx = torch.cat([idx0, idx1, idx], dim=0)

    ones = torch.ones(idx.shape[1], dtype=torch.bool, device=data.device)
    adjacency = torch.sparse_coo_tensor(
        idx,
        ones,
        (batch_size, num_points, num_points),
    ).to_dense()
    adjacency = (adjacency | adjacency.transpose(1, 2)).float()
    degree = torch.diag_embed(torch.sum(adjacency, dim=2))
    laplacian = degree - adjacency
    eigenvalues, eigenvectors = torch.linalg.eig(laplacian)
    return eigenvectors.real, laplacian, eigenvalues.real

