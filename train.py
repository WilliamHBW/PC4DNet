import time
import os
import torch
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
import MinkowskiEngine as ME

from utils import config_parser, write_parser
from Model.model import AutoEncoder
from Model.loss import *
from Dataset.data_loader import PCDataset, make_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.get_logger(os.path.join(config.train_savedir,'log.log'), logging.INFO)
        self.writer = SummaryWriter(log_dir=config.train_savedir)

        self.model = model.to(device)
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler(self.optimizer)
        self.logger.info(model)
        self.load_state_dict()
        self.epoch = 0
        self.record_set = {'bce':[], 'res_bpp':[], 'sum_loss':[], 'metrics':[], 'res':[]}
        self.best_loss = np.inf

    def get_logger(self, log_file, level):
        logger = logging.getLogger(__name__)
        logger.setLevel(level)
        logsh = logging.StreamHandler()
        logsh.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        logsh.setFormatter(formatter)
        logger.addHandler(logsh)

        # file log
        logfl = logging.FileHandler(log_file,mode="w", encoding="utf-8")
        logfl.setLevel(level)
        logfl.setFormatter(formatter)
        logger.addHandler(logfl)

        return logger

    def load_state_dict(self):
        """selectively load model
        """
        if self.config.loaddir==None:
            self.logger.info('Random initialization.')
            self.model.apply(self.weight_init)
        else:
            ckpt = torch.load(os.path.join(self.config.loaddir, 'best.pth'))
            self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.loaddir)
            
            self.optimizer.load_state_dict(ckpt['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.epoch = ckpt['epoch']
        return
         
    def save_checkpoint(self, model, iter, optimizer, scheduler, name):
        print('Saving model ...')
        torch.save({'epoch': iter, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, name)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, ME.MinkowskiConvolution):
            nn.init.kaiming_normal_(m.kernel.data, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, ME.MinkowskiBatchNorm):
            nn.init.constant_(m.bn.weight, 1)
            nn.init.constant_(m.bn.bias, 0)

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer
    
    def set_scheduler(self, optimizer):
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, verbose=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
        return scheduler

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items(): 
            if(k=='res'):
                continue
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items(): 
            if(k=='res'):
                continue
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))   
        for k, v in self.record_set.items(): 
            if(k=='res'):
                self.writer.add_histogram(k+'_'+main_tag, v[-1], self.epoch)
            else:
                self.writer.add_scalar(k+'_'+main_tag, np.mean(v), global_step)
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  

        return 
    
    
    @torch.no_grad()
    def valid(self, dataloader, main_tag='Test'):
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        for batch_step, coords in enumerate(tqdm(dataloader)):
            #data
            x = coords.to(device)
            x_len = coords.shape[0]
            
            #forward
            out_set = self.model(x, training=True)

            #cal loss
            bce_time, bce_list_time = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list_time'], out_set['ground_truth_list_time']):
                curr_bce = get_bce(out_cls, ground_truth)
                bce_time += curr_bce/len(out_set['out_cls_list_time'])
                bce_list_time.append(curr_bce.item())
            bce_space = []
            bce_list_space = []
            for i in range(args.frame_num):
                bce_space_i = 0
                bce_list_space_i = []
                for out_cls, ground_truth in zip(out_set['out_cls_list_space'][i], out_set['ground_truth_list_space'][i]):
                    curr_bce = get_bce(out_cls, ground_truth)
                    bce_space_i += curr_bce/len(out_set['out_cls_list_space'][i])
                    bce_list_space_i.append(curr_bce.item())
                bce_space.append(bce_space_i)
                bce_list_space.append(bce_list_space_i)
            
            bce = bce_time + sum(bce_space)/len(bce_space)
            bpp = get_bits(out_set['res_likelihood'])/float(float(x_len))
            sum_loss = self.config.alpha * bce + bpp

            metrics = [] 
            for out_cls, ground_truth in zip(out_set['out_cls_list_time'], out_set['ground_truth_list_time']):
                metrics.append(get_metrics(out_cls, ground_truth))
            for i in range(args.frame_num):
                for out_cls, ground_truth in zip(out_set['out_cls_list_space'][i], out_set['ground_truth_list_space'][i]):
                    metrics.append(get_metrics(out_cls, ground_truth))

            res_mean = torch.abs(out_set['res_prior'].F).detach().cpu()
            # record
            self.record_set['bce'].append(bce.item())
            self.record_set['res_bpp'].append(bpp.item())
            self.record_set['sum_loss'].append(sum_loss.item())
            self.record_set['metrics'].append(metrics)
            self.record_set['res'].append(res_mean)
            torch.cuda.empty_cache()# empty cache.

        self.record(main_tag=main_tag, global_step=self.epoch)
        if(sum_loss.item() < self.best_loss and main_tag=='Test'):
            savename = os.path.join(self.config.train_savedir, 'best.pth')
            self.save_checkpoint(self.model, self.epoch, self.optimizer, self.scheduler, savename)
            self.best_loss = sum_loss.item()

        self.epoch += 1

        return
        
    def train(self, dataloader, main_tag='Train'):
        #adjust RD coefficient
        if(self.epoch < 5):
            alpha = 20
        else:
            alpha = self.config.alpha

        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        self.logger.info('alpha:' + str(round(alpha,2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))
        
        for batch_step, coords in enumerate(tqdm(dataloader)):
            #init
            global_step=self.epoch*len(dataloader)+batch_step
            self.optimizer.zero_grad()

            #data
            x = coords.to(device)
            x_len = coords.shape[0]
            
            #forward
            out_set = self.model(x, training=True)

            #cal loss
            bce_time, bce_list_time = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list_time'], out_set['ground_truth_list_time']):
                curr_bce = get_bce(out_cls, ground_truth)
                bce_time += curr_bce/len(out_set['out_cls_list_time'])
                bce_list_time.append(curr_bce.item())
            bce_space = []
            bce_list_space = []
            for i in range(args.frame_num):
                bce_space_i = 0
                bce_list_space_i = []
                for out_cls, ground_truth in zip(out_set['out_cls_list_space'][i], out_set['ground_truth_list_space'][i]):
                    curr_bce = get_bce(out_cls, ground_truth)
                    bce_space_i += curr_bce/len(out_set['out_cls_list_space'][i])
                    bce_list_space_i.append(curr_bce.item())
                bce_space.append(bce_space_i)
                bce_list_space.append(bce_list_space_i)
            
            bce = bce_time + sum(bce_space)/len(bce_space)
            bpp = get_bits(out_set['res_likelihood'])/float(x_len)
            sum_loss = alpha * bce + bpp
            sum_loss.backward()
            self.optimizer.step()

            # metric & record
            with torch.no_grad():
                metrics = []
                for out_cls, ground_truth in zip(out_set['out_cls_list_time'], out_set['ground_truth_list_time']):
                    metrics.append(get_metrics(out_cls, ground_truth))
                for i in range(args.frame_num):
                    for out_cls, ground_truth in zip(out_set['out_cls_list_space'][i], out_set['ground_truth_list_space'][i]):
                        metrics.append(get_metrics(out_cls, ground_truth))
                self.record_set['bce'].append(bce.item())
                self.record_set['res_bpp'].append(bpp.item())
                self.record_set['sum_loss'].append(sum_loss.item())
                self.record_set['metrics'].append(metrics)
                self.record_set['res'].append(0)
                if(self.epoch%self.config.log_freq==0 and batch_step==len(dataloader)):
                    self.record(main_tag=main_tag, global_step=global_step)
                    savename = os.path.join(self.config.train_savedir, 'epoch_' + str(self.epoch) + '.pth')
                    self.save_checkpoint(self.model, self.epoch, self.optimizer, self.scheduler, savename)
            torch.cuda.empty_cache()# empty cache.

        self.record(main_tag=main_tag, global_step=self.epoch*len(dataloader))
        self.scheduler.step()
        
        return

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    
    if(args.train_store_new):
        curr_time = time.gmtime(time.time())
        ckpt_savedir =  args.train_savedir + '/' + str(curr_time.tm_year)+'_'+str(curr_time.tm_mon)+'_'+str(curr_time.tm_mday)+'_'+str(curr_time.tm_hour)+'_'+str(curr_time.tm_min)+'_'+str(curr_time.tm_sec)
        if(not os.path.exists(ckpt_savedir)):
            os.mkdir(ckpt_savedir)
        args.train_savedir = os.path.join(ckpt_savedir)
    else:
        args.train_savedir = args.loaddir
    
    write_parser(args, args.train_savedir+'/config.cfg')

    model = AutoEncoder(args)
    trainer = Trainer(config=args, model=model)
    train_dataset = PCDataset(args.traindir, args.frame_num, 'h5', 'train', args.train_size_ratio)
    valid_dataset = PCDataset(args.traindir, args.frame_num, 'h5', 'valid', args.train_size_ratio)
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False, num_workers=args.num_workers)
    valid_dataloader = make_data_loader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=args.num_workers)

    #train for end-to-end optimization
    for epoch in range(args.iter):
        #train/valid
        trainer.train(train_dataloader)
        trainer.valid(valid_dataloader)
