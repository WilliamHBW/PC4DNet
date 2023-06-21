import os
import numpy as np 
import torch 
import MinkowskiEngine as ME

def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().cpu(), step.long().cpu() 
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    device = data.device
    data, ground_truth = data.cpu(), ground_truth.cpu()
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())

    return torch.Tensor(mask).bool().to(device)

def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)

def sort_spare_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    indices_sort = np.argsort(array2vector(sparse_tensor.C.cpu(), 
                                           sparse_tensor.C.cpu().max()+1))
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort], 
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0], 
                                         device=sparse_tensor.device)

    return sparse_tensor_sort


def sort_by_coor_sum(f, stride):
    xyz, feature = f.C, f.F
    maximum = xyz.max() + 1
    coor_sum = xyz[:, 0] * maximum * maximum * maximum \
               + xyz[:, 1] * maximum * maximum \
               + xyz[:, 2] * maximum \
               + xyz[:, 3]
    _, idx = coor_sum.sort()
    xyz_, feature_ = xyz[idx], feature[idx]
    f_ = ME.SparseTensor(feature_, coordinates=xyz_, tensor_stride=stride, device=f.device)
    return f_

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
    
def index_by_channel(point1, idx, K=3):
    B, N1, C = point1.size()
    _, N2, C, __ = idx.size()
    point1_ = point1.transpose(1, 2).reshape(-1, N1, 1)
    idx_ = idx.transpose(1, 2).reshape(-1, N2, K)
    knn_point1 = index_points(point1_, idx_).reshape(B, C, N2, K).transpose(1, 2)
    # knn_point1 = point1_[np.arange(B * C), idx_].transpose(1, 2)
    return knn_point1


def get_target_by_sp_tensor(out, target_sp_tensor):
    with torch.no_grad():
        def ravel_multi_index(coords, step):
            coords = coords.long()
            step = step.long()
            coords_sum = coords[:, 0] \
                         + coords[:, 1] * step \
                         + coords[:, 2] * step * step \
                         + coords[:, 3] * step * step * step
            return coords_sum

        step = max(out.C.max(), target_sp_tensor.C.max()) + 1

        out_sp_tensor_coords_1d = ravel_multi_index(out.C, step)
        in_sp_tensor_coords_1d = ravel_multi_index(target_sp_tensor.C, step)

        # test whether each element of a 1-D array is also present in a second array.
        target = torch.isin(out_sp_tensor_coords_1d, in_sp_tensor_coords_1d)

    # return torch.Tensor(target).bool()
    return target