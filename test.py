import time
import os 
import MinkowskiEngine as ME
import numpy as np
import torch
import open3d as o3d
import pandas as pd

from utils import config_parser, write_parser, load_ply_sparse, pc_error
from Model.model import AutoEncoder
from Coder.coder import Coder
from Dataset.data_utils import static_pcc, write_ply_ascii_geo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

@torch.no_grad()
def test(args, model, filegroup):
    #results
    results = {}
    framename_first = ''
    bpps = []
    enc_time = []
    dec_time = []
    pc_error_metrics = []
    x_points = []
    x_dec_points = []

    #prepare
    ori_x_buffer = []
    for i in range(args.frame_num):
        pc = o3d.io.read_point_cloud(filegroup[i])
        pc_points = np.array(pc.points)
        coords = np.hstack((pc_points, np.zeros((pc_points.shape[0],1))+i))
        x_points.append(coords.shape[0])
        ori_x_buffer.append(load_ply_sparse(coords, device))
    
    #test
    framename = filegroup[0].split('/')[-1].split('.')[0]
    file_prefix = args.test_savedir+'/'+framename
    resfix = '_res'
        
    coder = Coder(model, file_prefix, args.frame_num)

    #encode
    print("="*10,"using 4D PCC")
    print("="*10,"Encoding")
    start_time = time.process_time()
    x = torch.concatenate(ori_x_buffer)
    _ = coder.encode(x, resfix=resfix)
    time_enc = round(time.process_time() - start_time, 3)

    # decode
    print("="*10,"Decoding")
    start_time = time.time()
    x_dec = coder.decode(resfix=resfix)
    time_dec = round(time.time() - start_time, 3)

    #bit rate
    print("="*10,"Evaluating")
    res_f_bits_ = np.array([os.path.getsize(file_prefix + resfix + postfix)*8 \
                            for postfix in ['_F.bin', '_H.bin', '_num_points.bin']])
    res_c_bits_ = 0
    for i in range(args.frame_num):
        res_c_bits_ += sum([os.path.getsize(file_prefix + resfix + '_' + str(i) + postfix)*8 for postfix in ['_C.bin']])
    bpps_ = ((sum(res_f_bits_)+res_c_bits_)/sum(x_points)).round(5)
     
    bpps.append(bpps_)

    #coding time
    enc_time.append(time_enc)
    dec_time.append(time_dec)

    #distortion
    for i in range(args.frame_num):
        x_dec_points.append(x_dec[i].C.detach().cpu().numpy()[:,1:].shape[0])
        write_ply_ascii_geo(file_prefix + '_rec_'+str(i)+'.ply', x_dec[i].C.detach().cpu().numpy()[:,1:])
        pc_error_metrics_ = pc_error(infile1=filegroup[i], infile2=file_prefix+'_rec_'+str(i)+'.ply', res=args.res, normal=False)
        pc_error_metrics.append(pc_error_metrics_)

    #average
    d1_psnr = []
    d1_mse = []
    for item in pc_error_metrics:
        d1_psnr.append(item["mseF,PSNR (p2point)"][0])
        d1_mse.append(item["mseF      (p2point)"][0])
    
    results['frame_name'] = framename_first
    results["num_points(input)"] = x_points
    results["num_points(output)"] = x_dec_points
    results["ave_bpps"] = bpps
    results["time(enc)"] = enc_time
    results["time(dec)"] = dec_time
    results["p2p_psnr"] = d1_psnr
    results["p2p_mse"] = d1_mse
    return results

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()

    loadname = args.loaddir.split('/')[-1]
    savedir = os.path.join(args.test_savedir, loadname)
    if(not os.path.exists(savedir)):
        os.mkdir(savedir)
    args.test_savedir = savedir
    write_parser(args, args.test_savedir+'/config.cfg')

    model = AutoEncoder(args).to(device)
    ckpt = torch.load(os.path.join(args.loaddir, 'epoch_35.pth'))
    model.load_state_dict(ckpt['model'])

    test_filelist = os.listdir(args.testdir)
    test_filelist.sort()
    assert(len(test_filelist) >= args.total_frame_num)

    results_total = []
    for i in range(0, args.total_frame_num, args.frame_num):
        print("="*10,"test frame group:",str(int(i/args.frame_num)),"="*10)
        if(i+args.frame_num > len(test_filelist)):
            break
        filegroup = []
        
        for j in range(args.frame_num):
            filepath = os.path.join(args.testdir,test_filelist[i+j])
            filegroup.append(filepath)

        results = test(args, model, filegroup)
        results_total.append(results)

    #print result
    result_dynamic_ave = {}
    result_dynamic_ave['frame_name'] = 'dynamic_average'
    result_dynamic_ave["num_points(input)"] = 0
    result_dynamic_ave["num_points(output)"] = 0
    result_dynamic_ave["ave_bpps"] = 0
    result_dynamic_ave["time(enc)"] = 0
    result_dynamic_ave["time(dec)"] = 0
    result_dynamic_ave["p2p_psnr"] = 0
    result_dynamic_ave["p2p_mse"] = 0
    print("total frame num: ", args.total_frame_num, " frame group num: ", args.frame_num, " resolution: ", args.res, " lambda value: ", args.alpha, " load ckpt: ", args.loaddir)
    for item in results_total:

        result_dynamic_ave["num_points(input)"]  += np.average(item["num_points(input)"])
        result_dynamic_ave["num_points(output)"] += np.average(item["num_points(output)"])
        result_dynamic_ave["ave_bpps"]           += np.average(item["ave_bpps"])          
        result_dynamic_ave["time(enc)"]          += np.average(item["time(enc)"])         
        result_dynamic_ave["time(dec)"]          += np.average(item["time(dec)"])         
        result_dynamic_ave["p2p_psnr"]           += np.average(item["p2p_psnr"])         
        result_dynamic_ave["p2p_mse"]            += np.average(item["p2p_mse"])          

    result_dynamic_ave["num_points(input)"]  /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["num_points(output)"] /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["ave_bpps"]           /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["time(enc)"]          /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["time(dec)"]          /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["p2p_psnr"]           /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["p2p_mse"]            /= (args.total_frame_num / args.frame_num)

    print("average   num_points(input/output)     bpps       enc_time       dec_time        p2p_mse       p2p_psnr")
    print(result_dynamic_ave['frame_name'],"    ",  result_dynamic_ave['num_points(input)'],'/',result_dynamic_ave['num_points(output)'],"    ",
          result_dynamic_ave['ave_bpps'],"    ", result_dynamic_ave['time(enc)'],"    ",   result_dynamic_ave['time(dec)'],"    ",     
          result_dynamic_ave['p2p_mse'],"    ",result_dynamic_ave['p2p_psnr'])
    
    results_total.append(result_dynamic_ave)
    #record
    csv_name = os.path.join(args.test_savedir, 'result.csv')
    pd.DataFrame(results_total).to_csv(csv_name, index=False)

    #plot