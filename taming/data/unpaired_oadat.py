import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

import h5py


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ''' return example dict with keys: image1, image2, [file_path_, lbl2, ...]. 
        image1 and image2 are numpy ndarrays with shape [H, W, C] and dtype float32]'''
        example = self.data[i]
        return example

class OADATUnpaired(CustomBase):
    def __init__(self, size, fname_h5_1, fname_h5_2, groupname_1, groupname_2, inds_1_start, inds_1_end, inds_2_start, inds_2_end, prng_seed=None):
        super().__init__()
        prng = np.random.RandomState(prng_seed)
        self.data = OADATDataset(size=size, fname_h5_1=fname_h5_1, fname_h5_2=fname_h5_2, groupname_1=groupname_1, groupname_2=groupname_2,\
                                 inds_1_start=inds_1_start, inds_1_end=inds_1_end, inds_2_start=inds_2_start, inds_2_end=inds_2_end, prng=prng)

class OADATSingle(CustomBase):
    def __init__(self, size, fname_h5, groupname, inds_start, inds_end, prng_seed=None):
        super().__init__()
        prng = np.random.RandomState(prng_seed)
        self.data = OADATDatasetSingleImage(size=size, fname_h5=fname_h5, groupname=groupname, inds_start=inds_start, inds_end=inds_end, prng=prng)


class OADATDataset(Dataset):
    def __init__(self, size, fname_h5_1, fname_h5_2, groupname_1, groupname_2, inds_1_start, inds_1_end, inds_2_start, inds_2_end, prng=None):
        self.fname_h5_1 = fname_h5_1
        self.fname_h5_2 = fname_h5_2
        self.groupname_1 = groupname_1
        self.groupname_2 = groupname_2
        self.inds_1_start = inds_1_start
        self.inds_1_end = inds_1_end
        self.inds_2_start = inds_2_start
        self.inds_2_end = inds_2_end
        self.prng = prng
        self.size = size
        self.check_data()
        if 'labels' in self.groupname_1:
            self.transform1 = self._scale_labels_to_minus_plus_1_fn
        else:
            self.transform1 = self._scaleclip_normalize_fn
        if 'labels' in self.groupname_2:
            self.transform2 = self._scale_labels_to_minus_plus_1_fn
        else:
            self.transform2 = self._scaleclip_normalize_fn

    def check_data(self):
        with h5py.File(self.fname_h5_1, 'r') as fh:
            self.len1_max = fh[self.groupname_1].shape[0]
        with h5py.File(self.fname_h5_2, 'r') as fh:
            self.len2_max = fh[self.groupname_2].shape[0]
        assert self.inds_1_start >= 0 and self.inds_1_end <= self.len1_max
        assert self.inds_2_start >= 0 and self.inds_2_end <= self.len2_max
        self.len1 = self.inds_1_end - self.inds_1_start
        self.len2 = self.inds_2_end - self.inds_2_start
        
    def __len__(self):
        return min(self.len1, self.len2)

    def _scale_labels_to_minus_plus_1_fn(self, x, max_lbl=2):
        '''Scale labels to [-1, 1] '''
        x = np.asarray(x, dtype=np.float32)
        x /= float(max_lbl)
        x *= 2
        x -= 1
        return x
    
    def _scaleclip_normalize_fn(self, x):
        '''Apply scaleclip, then normalize to [-1, 1] '''
        x = np.clip(x/np.max(x), a_min=-0.2, a_max=None)
        x -= np.min(x)
        x /= np.max(x)
        x *= 2
        x -= 1
        return x

    def __getitem__(self, index):
        i1 = self.inds_1_start + index  
        if i1 >= self.inds_1_end:
            while i1 >= self.inds_1_end:
                i1 -= self.len1
        if self.prng is not None:
            i2 = self.inds_2_start + index 
            if i2 >= self.inds_2_end:
                while i2 >= self.inds_2_end:
                    i2 -= self.len2
        else:
            i2 = np.random.randint(self.inds_2_start, self.inds_2_end)
        with h5py.File(self.fname_h5_1, 'r') as fh:
            img1 = fh[self.groupname_1][i1,...]
        with h5py.File(self.fname_h5_2, 'r') as fh:
            img2 = fh[self.groupname_2][i2,...]
        
        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        sample = {'image1': img1, 'image2': img2}
        return sample
    
class OADATDatasetSingleImage(Dataset):
    def __init__(self, size, fname_h5, groupname,inds_start, inds_end, prng=None):
        self.fname_h5 = fname_h5
        self.groupname = groupname
        self.inds_start = inds_start
        self.inds_end = inds_end
        self.prng = prng
        self.size = size
        self.check_data()
        if 'labels' in self.groupname:
            self.transform = self._scale_labels_to_minus_plus_1_fn
        else:
            self.transform = self._scaleclip_normalize_fn
        
    def check_data(self):
        with h5py.File(self.fname_h5, 'r') as fh:
            self.len_max = fh[self.groupname].shape[0]
        assert self.inds_start >= 0 and self.inds_end <= self.len_max
        self.len = self.inds_end - self.inds_start
        
    def __len__(self):
        return self.len

    def _scale_labels_to_minus_plus_1_fn(self, x, max_lbl=2):
        '''Scale labels to [-1, 1] '''
        x = np.asarray(x, dtype=np.float32)
        x /= float(max_lbl)
        x *= 2
        x -= 1
        return x
    
    def _scaleclip_normalize_fn(self, x):
        '''Apply scaleclip, then normalize to [-1, 1] '''
        x = np.clip(x/np.max(x), a_min=-0.2, a_max=None)
        x -= np.min(x)
        x /= np.max(x)
        x *= 2
        x -= 1
        return x

    def __getitem__(self, index):
        if self.prng is not None:
            i1 = np.random.randint(self.inds_start, self.inds_end)
        else:
            i1 = self.inds_start + index  
            if i1 >= self.inds_end:
                while i1 >= self.inds_end:
                    i1 -= self.len
        with h5py.File(self.fname_h5, 'r') as fh:
            img1 = fh[self.groupname][i1,...]
        
        img1 = self.transform(img1)
        sample = {'image': img1}
        return sample
    
    