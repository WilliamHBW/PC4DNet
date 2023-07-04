import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from Model.model_utils import *

class InceptionResNet(torch.nn.Module):
    #IRN block

    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x

        return out
    
def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))
        
    return torch.nn.Sequential(*layers)

class DownSample(torch.nn.Module):
    def __init__(self, ic0,oc0,oc1,scale):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels = ic0,
            out_channels = oc0, 
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 3
        )
        self.down0 = ME.MinkowskiConvolution(
            in_channels = oc0,
            out_channels = oc1,
            kernel_size = 2,
            stride = scale,
            bias = True,
            dimension = 3
        )
        self.block0 = make_layer(
            block = InceptionResNet,
            block_layers = 3,
            channels = oc1
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
    
    def forward(self, x):
        out0 = self.relu(self.down0(self.relu(self.conv0(x))))
        out0 = self.block0(out0)
        return out0

class UpSample(torch.nn.Module):
    def __init__(self,ic0,oc0,oc1,scale):
        super().__init__()
        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels = ic0,
            out_channels = oc0,
            kernel_size = 2,
            stride = scale,
            bias = True,
            dimension = 3
        )
        self.conv0 = ME.MinkowskiConvolution(
            in_channels = oc0,
            out_channels = oc1,
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 3
        )
        self.block0 = make_layer(
            block = InceptionResNet,
            block_layers = 3,
            channels = oc1
        )
        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels = oc1,
            out_channels = 1,
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 3
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training:
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))
        return data_pruned
    
    def forward(self, x, nums, ground_truth, training=True):
        out = self.relu(self.up0(x))
        out = self.relu(self.conv0(out))
        out = self.block0(out)
        out_cls = self.conv0_cls(out)
        out = self.prune_voxel(out, out_cls, nums, ground_truth, training)
        return out_cls, out

   
class ResNet(torch.nn.Module):
    """
    Basic block: Residual
    """

    def __init__(self, channels):
        super(ResNet, self).__init__()
        # path_1
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out
    


class InceptionResNet_4D(torch.nn.Module):
    #IRN block

    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=4)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=4)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=4)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=4)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=4)

        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x

        return out


class ResNet_4D(torch.nn.Module):
    """
    Basic block: Residual
    """

    def __init__(self, channels):
        super(ResNet, self).__init__()
        # path_1
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=4)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=4)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


class DownSample_4D(torch.nn.Module):
    def __init__(self, ic0,oc0,oc1,scale,temporal_kernel_size):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels = ic0,
            out_channels = oc0, 
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 4
        )
        kernel_generator = ME.KernelGenerator(kernel_size=temporal_kernel_size,stride=scale,region_type=ME.RegionType.HYPER_CROSS,dimension=4)
        self.down0 = ME.MinkowskiConvolution(
            in_channels = oc0,
            out_channels = oc1,
            kernel_size = temporal_kernel_size,
            stride = scale,
            bias = True,
            kernel_generator = kernel_generator,
            dimension = 4
        )
        self.block0 = make_layer(
            block = InceptionResNet_4D,
            block_layers = 3,
            channels = oc1
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
    
    def forward(self, x):
        out0 = self.relu(self.down0(self.relu(self.conv0(x))))
        out0 = self.block0(out0)
        return out0


class UpSample_4D(torch.nn.Module):
    def __init__(self,ic0,oc0,oc1,scale,temporal_kernel_size):
        super().__init__()
        kernel_generator = ME.KernelGenerator(kernel_size=temporal_kernel_size,stride=scale,region_type=ME.RegionType.HYPER_CROSS,dimension=4)
        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels = ic0,
            out_channels = oc0,
            kernel_size = temporal_kernel_size,
            stride = scale,
            bias = True,
            kernel_generator = kernel_generator,
            dimension = 4
        )
        self.conv0 = ME.MinkowskiConvolution(
            in_channels = oc0,
            out_channels = oc1,
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 4
        )
        self.block0 = make_layer(
            block = InceptionResNet_4D,
            block_layers = 3,
            channels = oc1
        )
        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels = oc1,
            out_channels = 1,
            kernel_size = 3,
            stride = 1,
            bias = True,
            dimension = 4
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training:
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))
        return data_pruned
    
    def forward(self, x, nums, ground_truth, training=True):
        out = self.relu(self.up0(x))
        out = self.relu(self.conv0(out))
        out = self.block0(out)
        out_cls = self.conv0_cls(out)
        out = self.prune_voxel(out, out_cls, nums, ground_truth, training)
        return out_cls, out