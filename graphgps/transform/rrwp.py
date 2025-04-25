import torch
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor


@torch.no_grad()
def calculate_rrwp(data,
                   k_steps=8):
    """Calculate the Relative Random Walk Probabilities (RRWP) matrix for a given number of k-steps."""
    
    # Calculate the adjacency matrix
    A = to_dense_adj(data.edge_index)[0]

    # Compute transition matrix M = D^(-1) * A
    # where D is the degree matrix
    degree_vector = A.sum(dim=1)
    inv_degree_vector = 1.0 / degree_vector
    inv_degree_vector[inv_degree_vector == float('inf')] = 0.0 # Avoid division by zero
    D_inv = torch.diag(inv_degree_vector)

    M = torch.matmul(D_inv, A)

    # Compute powers of M : [I, M, M^2, ..., M^k]
    M_powers = [torch.eye(data.num_nodes, dtype=torch.float)]
    for k in range(1, k_steps):
        M_powers.append(torch.linalg.matrix_power(M, k))

    # Stack the powers into a tensor n x n x k
    P = torch.stack(M_powers, dim=-1)

    # Use Torch.sparse to create a sparse tensor for the RRWP matrix
    P_sparse = SparseTensor.from_dense(P, has_value=True)

    # Retrieve the indices and values of the sparse tensor
    P_rows, P_cols, P_vals = P_sparse.coo()
    P_indices = torch.stack([P_rows, P_cols], dim=0)

    # Retrieve the diagonal elements (n x k) to use as initial node-level structural encodings (same as RWSE)
    P_diag = torch.diagonal(P).transpose(0, 1)

    data["rrwp"] = P_diag
    data["rrwp_index"] = P_indices
    data["rrpw_value"] = P_vals

    return data