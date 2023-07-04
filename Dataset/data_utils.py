from plyfile import PlyData, PlyElement
import open3d as o3d
import numpy as np
import h5py
import subprocess
import os
import torch

def write_ply_ascii_geo(filedir, points, text=True): 
    points_v = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points_v, dtype=[('x','f4'),('y','f4'),('z','f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el],text=text).write(filedir)


def read_ply_ascii_geo(filedir):
    ply = o3d.io.read_point_cloud(filedir)
    points = np.array(ply.points)
    colors = np.array(ply.colors)
    return points, colors

def read_h5_geo(filedir, frame_num):
    pc = h5py.File(filedir, 'r')
    coord = pc.get('data')[()]

    return coord

def write_h5_geo(filedir, coords_list):
    frame_num = len(coords_list)
    coords = np.concatenate(coords_list)
    with h5py.File(filedir, 'w') as h:
        data = coords.astype('short')
        h.create_dataset('data', data=data, shape=data.shape)

    return

def get_logger_sh(subp, log_f=None, show=False):
    c=subp.stdout.readline()
    lines = []
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show:
            print(line)
        if log_f != None:
            log_f.write(line)
        lines.append(line)
        c=subp.stdout.readline()
    return lines

#change working dir
class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    def __exit__(self, etype,value, traceback):
        os.chdir(self.savedPath)

pcgcdir = '/home/mmspg/Desktop/PCGCv2'
def static_pcc(points_geo, filepath, ckptdir, qp, show=False):
    tmppath = filepath
    tmpname = filepath.split('/')[-1].split('.')[0]
    write_ply_ascii_geo(tmppath, points_geo)
    res = str(2**(int(qp)))
    command = 'python coder.py --filedir='+tmppath+' --ckptdir='+ckptdir+' --scaling_factor=1.0 --rho=1.0 --res='+res

    with cd(pcgcdir):
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        return_code = subp.wait()
        assert(return_code==0)
        _ = get_logger_sh(subp,show=show)
    
    read_path = '/home/mmspg/Desktop/PCGCv2/output/'+tmpname+'_dec.ply'
    ply_dec = o3d.io.read_point_cloud(read_path)
    points_dec = np.asarray(ply_dec.points)
    os.system('rm '+tmppath)
    return points_dec


def visualize_pc_features(pc1, batch_id, name):
    #input is ME sparsetensor
    coords = pc1.coordinates_at(batch_id)
    feats = pc1.features_at(batch_id)
    N,C = feats.size()
    max_val = 255
    min_val = 0
    data = np.empty(N,dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    data['x'] = coords[:,0].cpu().numpy()
    data['y'] = coords[:,1].cpu().numpy()
    data['z'] = coords[:,2].cpu().numpy()

    #create seperate pc for each channel
    for i in range(C):
        feat = feats[:,i]
        feat_min = torch.min(feat)
        feat_max = torch.max(feat)
        feat_rescale = min_val + (feat - feat_min) * (max_val - min_val) / (feat_max - feat_min)

        if(name=='test'):
            data['red'] = (np.ones_like(feat_rescale.round().cpu().numpy())*128).astype(np.uint8)
            data['green'] = (np.ones_like(feat_rescale.round().cpu().numpy())*128).astype(np.uint8)
            data['blue'] = (np.ones_like(feat_rescale.round().cpu().numpy())*128).astype(np.uint8)
        else:
            data['red'] = feat_rescale.round().cpu().numpy().astype(np.uint8)
            data['green'] = feat_rescale.round().cpu().numpy().astype(np.uint8)
            data['blue'] = feat_rescale.round().cpu().numpy().astype(np.uint8)

        vertex = PlyElement.describe(data, 'vertex')

        outputname = '/home/mmspg/Desktop/PCMCNet/Results/feature_out_' + name + '_' + str(i) + '.ply'

        PlyData([vertex]).write(outputname)