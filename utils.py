import configargparse
import numpy as np
import time
import MinkowskiEngine as ME
import subprocess
import pandas as pd

from Model.loss import *
from Dataset.data_utils import cd, get_logger_sh

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    
    #model
    parser.add_argument('--alpha', type=float, default=1, help='RD optimization weight, R + alpha x D')
    parser.add_argument('--loaddir', type=str, help='path to load init ckpt')
    parser.add_argument('--frame_num', type=int, default=2, help='size of frame group')
    #train
    parser.add_argument('--traindir', type=str, help='input train data path')
    parser.add_argument('--train_savedir', type=str, help='path to save train log info and train results')
    parser.add_argument('--train_store_new', action='store_true', help='if stored as a new checkpoint')
    parser.add_argument('--train_size_ratio', type=float, default=1.0, help='control the size of train dataset for fast verification')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, i.e. number of input file per gradient step')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--iter', type=int, default=20, help='training iteration')
    parser.add_argument('--log_freq', type=int, default=10, help='every log_freq turn save training info automatically')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')
    
    #test
    parser.add_argument('--testdir', type=str, help='input test data path')
    parser.add_argument('--test_savedir', type=str, help='path to save test log info and test results')
    parser.add_argument('--total_frame_num', type=int, default=8, help='total frame num for test')
    parser.add_argument('--res', type=int, default=1024, help='the spatial resolution of the input point cloud')

    return parser

def write_parser(args, savepath):
    argsDict = args.__dict__
    with open(savepath, 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

def load_ply_sparse(coords, device):
    coords = coords.astype('int')
    feats = np.ones([coords.shape[0],1]).astype('float')
    coords_list = [coords]
    feats_list = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_list, feats_list)
    x = ME.SparseTensor(features=feats_batch.float(), coordinates=coords_batch.int(), device=device)
    return x

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item) 
        except ValueError:
            continue
        
    return number

def pc_error(infile1, infile2, res, normal=False, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
               "h.       1(p2point)", "h.,PSNR  1(p2point)" ]

    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)", 
               "h.       2(p2point)", "h.,PSNR  2(p2point)" ]

    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)", 
               "h.        (p2point)", "h.,PSNR   (p2point)" ]

    haders_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                      "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                      "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headersF + haders_p2plane

    command = str('./pc_error_d' + 
                          ' --fileA='+infile1+ 
                          ' --fileB='+infile2+ 
                          ' --hausdorff=0 '+ 
                          ' --resolution='+str(res-1))
    
    #print(command)
    if normal:
      headers += haders_p2plane
      command = str(command + ' -n ' + infile1)

    results = {}
   
    start = time.time()
    with cd('/home/mmspg/Desktop/PCMCNet/Thirdparty'):
        subp=subprocess.Popen(command, 
                          shell=True, stdout=subprocess.PIPE)
        return_code = subp.wait()
        assert(return_code==0)
        lines = get_logger_sh(subp,show=show)
    
    for line in lines:
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
    # print('===== measure PCC quality using `pc_error` version 0.13.4', round(time.time() - start, 4))

    return pd.DataFrame([results])
