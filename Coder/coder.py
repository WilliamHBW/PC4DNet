from .coder_utils import *
from Model.model_utils import *
from Dataset.data_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """
    def __init__(self, filename):
        self.filename = filename
        self.ply_filename = filename + '.ply'

    def encode(self, coords, postfix=''):
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=self.ply_filename, points=coords)
        gpcc_encode(self.ply_filename, self.filename+postfix+'_C.bin')
        os.system('rm '+self.ply_filename)
        
        return 

    def decode(self, postfix=''):
        gpcc_decode(self.filename+postfix+'_C.bin', self.ply_filename)
        coords,_ = read_ply_ascii_geo(self.ply_filename)
        os.system('rm '+self.ply_filename)
        
        return coords

class FeatureCoder():
    """encode/decode feature using learned entropy model
    """
    def __init__(self, filename, entropy_model):
        self.filename = filename
        self.entropy_model = entropy_model.cpu()

    def encode(self, feats, postfix=''):
        strings, min_v, max_v = self.entropy_model.compress(feats.cpu())
        shape = feats.shape
        with open(self.filename+postfix+'_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(self.filename+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
            fout.write(np.array(min_v, dtype=np.float32).tobytes())
            fout.write(np.array(max_v, dtype=np.float32).tobytes())
            
        return 

    def decode(self, postfix=''):
        with open(self.filename+postfix+'_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename+postfix+'_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            max_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            
        feats = self.entropy_model.decompress(strings, min_v, max_v, shape, channels=shape[-1])
        
        return feats

class Coder():
    def __init__(self, model, filename):
        self.model = model 
        self.filename = filename
        self.coordinate_coder = CoordinateCoder(filename)
        self.feature_coder_sf = FeatureCoder(self.filename, model.entropy_bottleneck_sf)
        self.feature_coder_res = FeatureCoder(self.filename, model.entropy_bottleneck_res)
        self.down_conv = ME.MinkowskiConvolution(
            in_channels = 8,
            out_channels = 8,
            kernel_size = 2,
            stride = 2,
            bias = True,
            dimension = 3
        )
    
    @torch.no_grad()
    def encode(self, x, x_ref, resfix='', sffix=''):
        # Encoder
        y = self.model.feature_extract(x)
        y_ref = self.model.feature_extract(x_ref)
        #visualize_pc_features(y[0],0,'target')
        #visualize_pc_features(y_ref[0],0,'ref')
        res,_,_,_,sf_q,_,_,_ = self.model.encoder(x, y, y_ref, training=False)
        res_ = sort_spare_tensor(res)
        sf_q = sort_by_coor_sum(sf_q, 16)
        num_points = [len(ground_truth) for ground_truth in y[1:] + [x]]
        with open(self.filename+resfix+'_num_points.bin', 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
        self.feature_coder_res.encode(res_.F, postfix=resfix)
        self.coordinate_coder.encode((res_.C//res_.tensor_stride[0]).detach().cpu()[:,1:], postfix=resfix)
        self.feature_coder_sf.encode(sf_q.F, postfix=sffix)
        return sf_q, res_
    
    @torch.no_grad()
    def decode(self, x_ref, x, resfix='', sffix='', rho=1):
        y_ref = self.model.feature_extract(x_ref)
        y = self.model.feature_extract(x)
        # decode coords
        res_C = self.coordinate_coder.decode(postfix=resfix)
        res_C = torch.cat((torch.zeros((len(res_C),1)).int(), torch.tensor(res_C).int()),dim=-1)
        indices_sort = np.argsort(array2vector(res_C, res_C.max()+1))
        res_C = res_C[indices_sort]

        res_F = self.feature_coder_res.decode(postfix=resfix)
        res = ME.SparseTensor(features=res_F, coordinates=res_C*8, tensor_stride=8, device=device)

        with open(self.filename+resfix+'_num_points.bin', 'rb') as fin:
            num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
            num_points[-1] = int(rho * num_points[-1])# update
            num_points = [[num] for num in num_points]
        
        sf_F = self.feature_coder_sf.decode(postfix=sffix)
        down_model = self.down_conv.to(device)
        y4 = down_model(res)
        sf = ME.SparseTensor(features=sf_F, coordinate_map_key=y4.coordinate_map_key, coordinate_manager=y4.coordinate_manager, device=device)
        
        _,_,out = self.model.decoder(y_ref[0], sf, res, nums_list=num_points, ground_truth_list=[None]*3, training=False)

        return out
