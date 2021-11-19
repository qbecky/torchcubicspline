import numpy as np
import torch

torch.set_default_dtype(torch.float64)

def cheap_stack(tensors, dim):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)

def expand_to_target_size(tensor, target_size):
    '''
    Input:
    - tensor      : a tensor of shape (?,) to expand
    - target_size : a tensor size

    Ouptut:
    - exp_tensor : a tensor of shape (?, *target_size)
    '''
    # new_shape gives the number of new dimensions to add to a column vector of 'tensor' to match targeted order
    new_shape   = [tensor.shape[0]] + [1 for i in range(len(target_size)-1)]
    # repeat_list gives how many times each dimension should be repeated so that 'tensor' has the correct shape
    repeat_list = [1] + list(target_size)
    exp_tensor = tensor.reshape(*new_shape).repeat(*repeat_list)
    return exp_tensor

def bucketize_different_boundaries(tensor, boundaries): 
    '''
    This function extends torch.bucketize in the case of boundaries with 

    Input:
    - tensor : a tensor of shape (?, d1, ..., dn) to be bucketized, the first dimension is the batch dimension
    - boundaries : a tensor of shape (?, n_bdry) containing 
    
    Output:
    - result : a tensor of shape (?, d1, ..., dn) containing the bucket ids
    ''' 

    results = torch.zeros_like(tensor, dtype=torch.int64)
    # essentially the same thing as expand_to_target_size, enables reusing new_shape and repeat_list
    new_shape   = [boundaries.shape[0]] + [1 for i in range(len(tensor.shape)-1)]
    repeat_list = [1] + list(tensor.shape[1:])
    for j in range(boundaries.shape[1]):
        reshaped_boundary = boundaries[:, j].reshape(*new_shape).repeat(*repeat_list)
        results += (tensor > reshaped_boundary).int()
    return results


def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    """Solves a tridiagonal system Ax = b.

    The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
    and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
    of size (k, k), with entries:

    D[0] U[0]
    L[0] D[1] U[1]
         L[1] D[2] U[2]                     0
              L[2] D[3] U[3]
                  .    .    .
                       .      .      .
                           .        .        .
                        L[k - 3] D[k - 2] U[k - 2]
           0                     L[k - 2] D[k - 1] U[k - 1]
                                          L[k - 1]   D[k]

    Arguments:
        b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
        A_upper: A tensor of shape (..., k - 1).
        A_diagonal: A tensor of shape (..., k).
        A_lower: A tensor of shape (..., k - 1).

    Returns:
        A tensor of shape (..., k), corresponding to the x solving Ax = b

    Warning:
        This implementation isn't super fast. You probably want to cache the result, if possible.
    """

    # This implementation is very much written for clarity rather than speed.
    A_upper, _ = torch.broadcast_tensors(A_upper.unsqueeze(len(A_upper.shape)-1), b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower.unsqueeze(len(A_lower.shape)-1), b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal.unsqueeze(len(A_diagonal.shape)-1), b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)
