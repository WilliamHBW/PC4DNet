import torch
import MinkowskiEngine as ME
import sys 
import math

from Model.block import *
from Model.model_utils import *
sys.path.append('/home/mmspg/Desktop/PC4DNet/')
from Coder import EntropyBottleneck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Feature_extract(torch.nn.Module):
    def __init__(self, dscale=2, channels=[1,16,32,64]):
        super().__init__()
        self.down_block0 = DownSample(channels[0], channels[1], channels[2], dscale)
        self.down_block1 = DownSample(channels[2], channels[2], channels[3], dscale)
       
    def forward(self, x):
        out0 = self.down_block0(x)
        out1 = self.down_block1(out0)
        return [out1, out0]

class Recons(torch.nn.Module):
    def __init__(self, uscale=2, channels=[64,32,16]):
        super().__init__()
        self.up_block0 = UpSample(channels[0], channels[1], channels[1], uscale)
        self.up_block1 = UpSample(channels[1], channels[2], channels[2], uscale)

    def forward(self, x, nums_list, ground_truth_list,training=True):
        out_cls_0, out = self.up_block0(x,   nums_list[0], ground_truth_list[0], training)
        out_cls_1, out = self.up_block1(out, nums_list[1], ground_truth_list[1], training)

        out_cls_list = [out_cls_0, out_cls_1]
        return out_cls_list, out

class Temporal_Encoder(torch.nn.Module):
    def __init__(self, dscale, channels=[64,32,8], temporal_kernel_size=[2,2,2,1]):
        super().__init__()
        self.down_block0_4d = DownSample_4D(channels[0], channels[0], channels[1], dscale, temporal_kernel_size)
        self.conv0_4d = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=4)
        self.relu = ME.MinkowskiReLU(inplace=True)
    
    def forward(self,x):
        out = self.down_block0_4d(x)
        out = self.relu(self.conv0_4d(out))
        return out

class Temporal_Decoder(torch.nn.Module):
    def __init__(self, uscale, channels=[8,32,64], temporal_kernel_size=[2,2,2,1]):
        super().__init__()
        self.up_block0_4d = UpSample_4D(channels[1], channels[2], channels[2], uscale, temporal_kernel_size)
        self.conv0_4d = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=4)
        self.relu = ME.MinkowskiReLU(inplace=True)
    
    def forward(self,x, nums_list, ground_truth_list, training=True):
        out = self.relu(self.conv0_4d(x))
        out_cls, out = self.up_block0_4d(out, nums_list[0], ground_truth_list[0], training)
        return out_cls, out

    
class AutoEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.frame_num = args.frame_num

        self.spatial_encoder = Feature_extract(dscale=2, channels=[1,16,32,64])
        self.spatial_decoder = Recons(uscale=2, channels=[64,32,16])
        self.temporal_encoder = Temporal_Encoder(dscale=[2,2,2,1], channels=[64,32,8], temporal_kernel_size=[2,2,2,3])
        self.temporal_decoder = Temporal_Decoder(uscale=[2,2,2,1], channels=[8,32,64], temporal_kernel_size=[2,2,2,3])
        self.entropy_bottleneck = EntropyBottleneck(8)

    def get_likelihood(self, data, quantize_mode, entropy_model):
        data_F, likelihood = entropy_model(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)
        
        return data_Q, likelihood
    
    def encoder(self, x, training=True):
        ground_truth_list_space = []
        nums_list_space = []
        coords = torch.zeros((1,5)).to(device)
        feats = torch.zeros((1,64)).to(device)
        for i in range(self.frame_num):
            x_coord = x[x[:,-1]==i][:,:-1].contiguous()
            x_feat = torch.ones((x_coord.shape[0],1)).to(device)
            x_space = ME.SparseTensor(features=x_feat.float(), coordinates=x_coord.int(), tensor_stride=1, device=device)
            y_space_i = self.spatial_encoder(x_space)

            ground_truth_list_i = y_space_i[1:] + [x_space]
            nums_list_i = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list_i]

            ground_truth_list_space.append(ground_truth_list_i)
            nums_list_space.append(nums_list_i)

            y_space_coord = torch.hstack((y_space_i[0].C, (torch.zeros((y_space_i[0].C.shape[0],1))+i).to(device)))
            coords = torch.vstack((coords, y_space_coord))
            feats = torch.vstack((feats, y_space_i[0].F))
        coords = coords[1:]
        feats = feats[1:]
        y_space_merged = ME.SparseTensor(coordinates=coords.int(), features=feats.float(), tensor_stride=[4,4,4,1], device=device)
        
        ground_truth_list_time = [y_space_merged]
        nums_list_time = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list_time]

        y_time = self.temporal_encoder(y_space_merged)

        return y_time, ground_truth_list_space, ground_truth_list_time, nums_list_space, nums_list_time
    
    def decoder(self, y, nums_list_space, nums_list_time, ground_truth_list_space, ground_truth_list_time, training=True):
        y_space_cls, y_space = self.temporal_decoder(y, [nums_list_time[0]], [ground_truth_list_time[0]], training)

        x_cls = []
        x = []
        for i in range(self.frame_num):
            y_space_coord = y_space.C[y_space.C[:,-1]==i]
            y_space_feat = y_space.F[y_space.C[:,-1]==i]
            y_space_i = ME.SparseTensor(features=y_space_feat, coordinates=y_space_coord[:,:-1].contiguous(), tensor_stride=y_space.tensor_stride[0], device=device)
            out_cls, out = self.spatial_decoder(y_space_i, nums_list_space[i], ground_truth_list_space[i], training)

            x_cls.append(out_cls)
            x.append(out)
        x_cls_time = [y_space_cls]
        return x_cls_time, x_cls, x
    
    def forward(self, x, training=True):
        #with torch.autograd.profiler.profile(enabled=True,use_cuda=True) as prof:
        y, ground_truth_list_space, ground_truth_list_time, nums_list_space, nums_list_time = self.encoder(x)

        y_ = sort_sparce_tensor(y)
        y_q, y_likelihood = self.get_likelihood(y_, quantize_mode="noise" if training else "symbols", entropy_model=self.entropy_bottleneck)

        x_out_cls_list_time, x_out_cls_list_space, x_out = self.decoder(y_q, nums_list_space, nums_list_time, ground_truth_list_space, ground_truth_list_time, training) 
        #print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        return {'out':x_out,
                'out_cls_list_space':x_out_cls_list_space,
                'out_cls_list_time':x_out_cls_list_time,
                'res_prior':y_q, 
                'res_likelihood':y_likelihood, 
                'ground_truth_list_space':ground_truth_list_space,
                'ground_truth_list_time':ground_truth_list_time}
        
