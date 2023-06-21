import time
import os 
import MinkowskiEngine as ME
import numpy as np
import torch
import open3d as o3d
import pandas as pd

from utils import config_parser, write_parser, load_ply_sparse, pc_error
from Model.model import PCMCNet
from Coder.coder import Coder
from Dataset.data_utils import static_pcc, write_ply_ascii_geo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

@torch.no_grad()
def test(args, model, filegroup):
    #results
    results = {}
    framename_first = ''
    res_bits = []
    sf_bits = []
    bpps = []
    enc_time = []
    dec_time = []
    pc_error_metrics = []
    x_points = []
    x_dec_points = []

    #prepare
    ori_x_buffer = []
    rec_ply_buffer = []
    for i in range(args.frame_num):
        pc = o3d.io.read_point_cloud(filegroup[i])
        x_points.append(np.array(pc.points).shape[0])
        ori_x_buffer.append(load_ply_sparse(np.array(pc.points), device))
    
    #test
    for i in range(args.frame_num):
        print("="*10,"test frame num:",str(i),"="*10)
        framename = filegroup[i].split('/')[-1].split('.')[0]
        file_prefix = args.test_savedir+'/'+framename
        resfix = '_res'
        sffix = '_sf'
        
        #use static pcc for the first frame
        if(i==0):
            print("="*10,"using static PCC",str(i))
            framename_first = framename
            ckptdir = '/home/mmspg/Desktop/PCGCv2/ckpts/r7_0.4bpp.pth'
            if(args.alpha>=3.0):
                ckptdir = '/home/mmspg/Desktop/PCGCv2/ckpts/r7_0.4bpp.pth'
            elif(args.alpha>=1.0):
                ckptdir = '/home/mmspg/Desktop/PCGCv2/ckpts/r5_0.25bpp.pth'
            elif(args.alpha<1.0):
                ckptdir = '/home/mmspg/Desktop/PCGCv2/ckpts/r4_0.15bpp.pth'

            print("="*10,"Encoding")
            x_dec = static_pcc(ori_x_buffer[i].C.detach().cpu().numpy()[:,1:], '/home/mmspg/Desktop/PCMCNet/Dataset/tmp.ply', ckptdir, qp='10', show=False)
            print("="*10,"Decoding")
            os.system('mv /home/mmspg/Desktop/PCGCv2/output/tmp_C.bin '+args.test_savedir+'/'+framename+'_C.bin')
            os.system('mv /home/mmspg/Desktop/PCGCv2/output/tmp_F.bin '+args.test_savedir+'/'+framename+'_F.bin')
            os.system('mv /home/mmspg/Desktop/PCGCv2/output/tmp_H.bin '+args.test_savedir+'/'+framename+'_H.bin')
            os.system('mv /home/mmspg/Desktop/PCGCv2/output/tmp_num_points.bin '+args.test_savedir+'/'+framename+'_num_points.bin')
            os.system('mv /home/mmspg/Desktop/PCGCv2/output/tmp_dec.ply '+args.test_savedir+'/'+framename+'_rec.ply')

            #add recon to rec buffer
            rec_ply_buffer.append(load_ply_sparse(x_dec, device))
            x_dec_points.append(x_dec.shape[0])
            #bit rate
            print("="*10,"Evaluating")
            res_bits_ = np.array([os.path.getsize(file_prefix + postfix)*8 \
                                    for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
            bpps_ = (sum(res_bits_)/ori_x_buffer[i].C.shape[0]).round(5)
            
            res_bits.append(sum(res_bits_))
            bpps.append(bpps_)
    
            #coding time
    
            #distortion
            pc_error_metrics_ = pc_error(infile1=filegroup[i], infile2=args.test_savedir+'/'+framename+'_rec.ply', res=args.res, normal=False)
            pc_error_metrics.append(pc_error_metrics_)
            continue

        coder = Coder(model, file_prefix)

        #encode
        print("="*10,"using dynamic PCC",str(i))
        print("="*10,"Encoding")
        start_time = time.process_time()
        _ = coder.encode(ori_x_buffer[i], rec_ply_buffer[i-1],resfix=resfix, sffix=sffix)
        time_enc = round(time.process_time() - start_time, 3)

        # decode
        print("="*10,"Decoding")
        start_time = time.time()
        x_dec = coder.decode(rec_ply_buffer[i-1], ori_x_buffer[i], resfix=resfix, sffix=sffix)
        time_dec = round(time.time() - start_time, 3)

        #update buffer
        rec_ply_buffer.append(load_ply_sparse(x_dec.C.detach().cpu().numpy(), device=device))

        #bit rate
        print("="*10,"Evaluating")
        res_bits_ = np.array([os.path.getsize(file_prefix + resfix + postfix)*8 \
                                for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
        sf_bits_ = np.array([os.path.getsize(file_prefix + sffix + postfix)*8 \
                                for postfix in ['_F.bin', '_H.bin']])
        bpps_ = ((sum(res_bits_)+sum(sf_bits_))/ori_x_buffer[i].C.shape[0]).round(5)
        
        res_bits.append(sum(res_bits_))
        sf_bits.append(sum(sf_bits_))
        bpps.append(bpps_)

        #coding time
        enc_time.append(time_enc)
        dec_time.append(time_dec)

        #distortion
        x_dec_points.append(x_dec.C.detach().cpu().numpy()[:,1:].shape[0])
        write_ply_ascii_geo(file_prefix + '_rec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
        pc_error_metrics_ = pc_error(infile1=filegroup[i], infile2=file_prefix+'_rec.ply', res=args.res, normal=False)
        pc_error_metrics.append(pc_error_metrics_)

    #average
    res_bits_ave = np.average(res_bits)
    sf_bits_ave = np.average(sf_bits)
    bpps_ave = np.average(bpps)
    enc_time_ave = np.average(enc_time)
    dec_time_ave = np.average(dec_time)
    d1_psnr = []
    d1_mse = []
    for item in pc_error_metrics:
        d1_psnr.append(item["mseF,PSNR (p2point)"][0])
        d1_mse.append(item["mseF      (p2point)"][0])
    
    results['frame_name'] = framename_first
    results["num_points(input)"] = x_points
    results["num_points(output)"] = x_dec_points
    results["ave_res_bits"] = res_bits
    results["ave_sf_bits"] = sf_bits
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

    model = PCMCNet(args).to(device)
    ckpt = torch.load(os.path.join(args.loaddir, 'best.pth'))
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
    result_static_ave = {}
    result_static_ave['frame_name'] = 'static_average'
    result_static_ave["num_points(input)"] = 0
    result_static_ave["num_points(output)"] = 0
    result_static_ave["ave_res_bits"] = 0
    result_static_ave["ave_sf_bits"] = 0
    result_static_ave["ave_bpps"] = 0
    result_static_ave["time(enc)"] = 0
    result_static_ave["time(dec)"] = 0
    result_static_ave["p2p_psnr"] = 0
    result_static_ave["p2p_mse"] = 0

    result_dynamic_ave = {}
    result_dynamic_ave['frame_name'] = 'dynamic_average'
    result_dynamic_ave["num_points(input)"] = 0
    result_dynamic_ave["num_points(output)"] = 0
    result_dynamic_ave["ave_res_bits"] = 0
    result_dynamic_ave["ave_sf_bits"] = 0
    result_dynamic_ave["ave_bpps"] = 0
    result_dynamic_ave["time(enc)"] = 0
    result_dynamic_ave["time(dec)"] = 0
    result_dynamic_ave["p2p_psnr"] = 0
    result_dynamic_ave["p2p_mse"] = 0
    print("total frame num: ", args.total_frame_num, " frame group num: ", args.frame_num, " resolution: ", args.res, " lambda value: ", args.alpha, " load ckpt: ", args.loaddir)
    for item in results_total:
        
        result_static_ave["num_points(input)"]  += item["num_points(input)"][0]
        result_static_ave["num_points(output)"] += item["num_points(output)"][0]
        result_static_ave["ave_res_bits"]       += item["ave_res_bits"][0]      
        result_static_ave["ave_bpps"]           += item["ave_bpps"][0]        
        result_static_ave["p2p_psnr"]           += item["p2p_psnr"][0]          
        result_static_ave["p2p_mse"]            += item["p2p_mse"][0]   

        result_dynamic_ave["num_points(input)"]  += np.average(item["num_points(input)"][1:])
        result_dynamic_ave["num_points(output)"] += np.average(item["num_points(output)"][1:])
        result_dynamic_ave["ave_res_bits"]       += np.average(item["ave_res_bits"][1:])     
        result_dynamic_ave["ave_sf_bits"]        += np.average(item["ave_sf_bits"][0:])      
        result_dynamic_ave["ave_bpps"]           += np.average(item["ave_bpps"][1:])          
        result_dynamic_ave["time(enc)"]          += np.average(item["time(enc)"][0:])         
        result_dynamic_ave["time(dec)"]          += np.average(item["time(dec)"][0:])         
        result_dynamic_ave["p2p_psnr"]           += np.average(item["p2p_psnr"][1:])         
        result_dynamic_ave["p2p_mse"]            += np.average(item["p2p_mse"][1:])          

    result_static_ave["num_points(input)"]  /= (args.total_frame_num / args.frame_num)
    result_static_ave["num_points(output)"] /= (args.total_frame_num / args.frame_num)
    result_static_ave["ave_res_bits"]       /= (args.total_frame_num / args.frame_num)
    result_static_ave["ave_sf_bits"]        /= (args.total_frame_num / args.frame_num)
    result_static_ave["ave_bpps"]           /= (args.total_frame_num / args.frame_num)
    result_static_ave["time(enc)"]          /= (args.total_frame_num / args.frame_num)
    result_static_ave["time(dec)"]          /= (args.total_frame_num / args.frame_num)
    result_static_ave["p2p_psnr"]           /= (args.total_frame_num / args.frame_num)
    result_static_ave["p2p_mse"]            /= (args.total_frame_num / args.frame_num)

    result_dynamic_ave["num_points(input)"]  /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["num_points(output)"] /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["ave_res_bits"]       /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["ave_sf_bits"]        /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["ave_bpps"]           /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["time(enc)"]          /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["time(dec)"]          /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["p2p_psnr"]           /= (args.total_frame_num / args.frame_num)
    result_dynamic_ave["p2p_mse"]            /= (args.total_frame_num / args.frame_num)

    print("average   num_points(input/output)     res_bits    sf_bits      bpps       enc_time       dec_time        p2p_mse       p2p_psnr")
    print(result_static_ave['frame_name'],"    ",  result_static_ave['num_points(input)'],'/',result_static_ave['num_points(output)'],"    ",
          result_static_ave['ave_res_bits'],"    ",result_static_ave['ave_sf_bits'],"    ",   result_static_ave['ave_bpps'],"    ",
          result_static_ave['time(enc)'],"    ",   result_static_ave['time(dec)'],"    ",     result_static_ave['p2p_mse'],"    ",result_static_ave['p2p_psnr'])
    
    print(result_dynamic_ave['frame_name'],"    ",  result_dynamic_ave['num_points(input)'],'/',result_dynamic_ave['num_points(output)'],"    ",
          result_dynamic_ave['ave_res_bits'],"    ",result_dynamic_ave['ave_sf_bits'],"    ",   result_dynamic_ave['ave_bpps'],"    ",
          result_dynamic_ave['time(enc)'],"    ",   result_dynamic_ave['time(dec)'],"    ",     result_dynamic_ave['p2p_mse'],"    ",result_dynamic_ave['p2p_psnr'])
    
    print('total_average',"    ",  (result_static_ave['num_points(input)']+result_dynamic_ave['num_points(input)'])/2,'/',(result_static_ave['num_points(output)']+result_dynamic_ave['num_points(output)'])/2,"    ",
          (result_static_ave['ave_res_bits']+result_dynamic_ave['ave_res_bits'])/2,"    ",result_dynamic_ave['ave_sf_bits'],"    ",   (result_static_ave['ave_bpps']+result_dynamic_ave['ave_bpps'])/2,"    ",
          result_dynamic_ave['time(enc)'],"    ",   result_dynamic_ave['time(dec)'],"    ",     (result_static_ave['p2p_mse']+result_dynamic_ave['p2p_mse'])/2,"    ",(result_static_ave['p2p_psnr']+result_dynamic_ave['p2p_psnr'])/2)
    
    results_total.append(result_static_ave)
    results_total.append(result_dynamic_ave)
    #record
    csv_name = os.path.join(args.test_savedir, 'result.csv')
    pd.DataFrame(results_total).to_csv(csv_name, index=False)

    #plot