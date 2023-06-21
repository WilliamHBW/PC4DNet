import open3d as o3d 
import numpy as np
import random
import os,time
from tqdm import tqdm
from data_utils import *
import subprocess

def generate_train_dataset(cfg_dic):
    #set filedir
    plydir = cfg_dic['data_dir']
    seq_name = cfg_dic['seq_name']
    cfg_name = cfg_dic['cfg_name']
    frame_num = int(cfg_dic['frame_num'])
    group_num = int(cfg_dic['group_num'])
    seq_names = seq_name.split(" ")
    outdir = '/home/mmspg/Desktop/PC4DNet/Dataset/train/'+cfg_name
    if(not os.path.exists(outdir)):
            os.mkdir(outdir)

    outdatadir = os.path.join(outdir, 'data')
    if(not os.path.exists(outdatadir)):
            os.mkdir(outdatadir)
    #record cfgs
    cfg_file = open(outdir+'/config.cfg','w')
    for k in cfg_dic:
        cfg_file.write(k+':'+cfg_dic[k])
        cfg_file.write('\n')
    cfg_file.close()

    for seq in seq_names:
        frame_cache = []
        frame_pcgc_cache = []
        frame_count = 0
        print("="*10,seq,"="*10)
        readdir = os.path.join(plydir, seq)
        files = os.listdir(readdir)
        files.sort()
        assert(frame_num<=len(files))
        for i in tqdm(range(frame_num)):
            frame_count = frame_count + 1
            readpath = os.path.join(readdir,files[i])
            plydata = o3d.io.read_point_cloud(readpath)

            #decrease bit precision from 11 to 8
            points = np.asarray(plydata.points).astype(np.int32)
            if(cfg_dic['quant_bit']=='9'):
                points_9bit = np.round(points / 4)
            elif(cfg_dic['quant_bit']=='8'):
                points_9bit = np.round(points / 8)
            else:
                points_9bit = np.round(points / 8)
            
            points_geo = np.unique(points_9bit, axis=0)
            frame_cache.append(points_geo)

            #write geo ply
            if(cfg_dic['out_filetype']=='h5'):
                if(frame_count>=group_num):
                    outname = seq + '_' + str(i) + '_' + str(group_num) + '.h5'
                    outpath = os.path.join(outdatadir, outname)
                    write_h5_geo(outpath, frame_cache)
                    frame_cache = frame_cache[1:]
            elif(cfg_dic['out_filetype']=='ply'):
                outname = seq + '_' + i + '.ply'
                outpath = os.path.join(outdatadir, outname)
                write_ply_ascii_geo(outpath, points_geo)

    return 

if  __name__ == '__main__':
    cfg_dic = {}
    cfg_dic['seq_name'] = 'exercise_vox11 basketball_player_vox11 dancer_vox11 model_vox11'
    #cfg_dic['seq_name'] = 'basketball_player_vox11'
    cfg_dic['data_dir'] = '/home/mmspg/Desktop/test_files/Owlii'
    cfg_dic['cfg_name'] = 'train1'
    cfg_dic['out_filetype'] = 'h5'
    cfg_dic['frame_num'] = '599'
    cfg_dic['group_num'] = '4' #frame group length
    cfg_dic['static_pcc'] = 'PCGCv2'
    cfg_dic['quant_bit'] = '9' #downsample precision
    generate_train_dataset(cfg_dic)