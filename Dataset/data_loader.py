from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import numpy as np 
import os
from Dataset.data_utils import read_h5_geo 
import MinkowskiEngine as ME
from tqdm import tqdm
import torch

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def collate_pointcloud_fn(list_data):
    coords_batch = ME.utils.batched_coordinates(list_data)

    return coords_batch

class PCDataset(Dataset):
    def __init__(self, filedir, frame_num=2, format='h5', mode='train', size_ratio=1):
        self.filedir = filedir
        self.cache = {}
        self.last_cache_percent = 0
        self.format = format
        self.frame_num = frame_num
        self.ratio = size_ratio
    
        assert mode in ('train','valid')

        filelist = os.listdir(filedir)
        flen = len(filelist)
        if(mode=='train'):
            self.files = filelist[0:int(np.floor(flen*0.95*self.ratio))]
        else:
            self.files = filelist[int(np.floor(flen*self.ratio*0.95)):int(flen*self.ratio)]
        self.len = len(self.files)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        if idx in self.cache:
            tensor = self.cache[idx]
        else:
            if self.format=='h5':
                coords_ = []
                filepath = os.path.join(self.filedir, self.files[idx])
                coords_ = read_h5_geo(filepath, self.frame_num)
            tensor = coords_
            #cache
            self.cache[idx] = tensor
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent

        return tensor

def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    } 
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = DataLoader(dataset, **args)

    return loader

from torch.utils.data import DataLoader
if __name__ == "__main__":
    filedir = '/home/mmspg/Desktop/PCMCNet/Dataset/train/train1/data'
    test_dataset = PCDataset(filedir, frame_num=8)
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=2, shuffle=True, num_workers=1, repeat=False)
   
    test_iter = iter(test_dataloader)
    print(test_iter)
    for i in tqdm(range(3)):
        coords, feats = next(test_iter)
        print("="*20, "check dataset", "="*20, 
            "\ncoords:\n", coords, "\nfeat:\n", feats)