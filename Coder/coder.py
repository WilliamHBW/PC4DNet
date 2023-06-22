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
    def __init__(self, model, filename, frame_num):
        self.model = model 
        self.filename = filename
        self.frame_num = frame_num
        self.coordinate_coder = CoordinateCoder(filename)
        self.feature_coder = FeatureCoder(self.filename, model.entropy_bottleneck)

    @torch.no_grad()
    def encode(self, x, resfix=''):
        # Encoder
        y, _,_, nums_list_space, nums_list_time = self.model.encoder(x, training=False)
        y_ = sort_sparce_tensor(y)

        num_points = nums_list_time
        for i in range(self.frame_num):
            num_points += nums_list_space[i]
        with open(self.filename+resfix+'_num_points.bin', 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())

        self.feature_coder.encode(y_.F, postfix=resfix)
        for i in range(self.frame_num):
            coord = y_.C[y_.C[:,-1]==i]
            self.coordinate_coder.encode((coord[:,1:-1]//y_.tensor_stride[0]).detach().cpu(), postfix=resfix+'_'+str(i))
        
        return
    
    @torch.no_grad()
    def decode(self, resfix='', rho=1):
        # decode coords
        y_C = torch.zeros((1,5))
        for i in range(self.frame_num):
            y_C_i = self.coordinate_coder.decode(postfix=resfix+'_'+str(i))
            y_C_i = torch.tensor(y_C_i) * 8
            y_C_i = torch.cat((torch.zeros((y_C_i.shape[0],1)).int(), y_C_i.int()),dim=-1)
            y_C_i = torch.hstack((y_C_i, (torch.zeros((y_C_i.shape[0],1))+i))).int()
            indices_sort = np.argsort(array2vector(y_C_i, y_C_i.max()+1))
            y_C_i = y_C_i[indices_sort]
            y_C = torch.vstack((y_C, y_C_i))
        y_C = y_C[1:]

        y_F = self.feature_coder.decode(postfix=resfix)
        y = ME.SparseTensor(features=y_F, coordinates=y_C.int(), tensor_stride=[8,8,8,1], device=device)

        with open(self.filename+resfix+'_num_points.bin', 'rb') as fin:
            num_points = np.frombuffer(fin.read(4*(1+2*self.frame_num)), dtype=np.int32).tolist()
            num_points[-1] = int(rho * num_points[-1])# update
            num_points = [[num] for num in num_points]
        nums_list_time = [num_points[0]]
        nums_list_space = []
        for i in range(self.frame_num):
            nums_list_space_i = num_points[1+i*2:3+i*2]
            nums_list_space.append(nums_list_space_i)

        _,_,out = self.model.decoder(y, nums_list_space, nums_list_time, ground_truth_list_space=[[None]*2]*self.frame_num, ground_truth_list_time=[None], training=False)

        return out
