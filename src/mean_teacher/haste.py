import torch
import numpy as np
import os
import pydicom
import pickle
import cv2
import random

PATH_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
PATH_MASK = os.path.join(PATH_ROOT, 'brainSeg/Code/')
PATH_LABEL = os.path.join(PATH_ROOT, 'iqa_data_source_rep/dataset_partition_combine/')
PATH_LABELED_DATA = os.path.join(PATH_ROOT, 'iqa_data_source_rep/reorganized_filtered_singleton')
PATH_UNLABELED_DATA = os.path.join(PATH_ROOT, 'iqa_data_source_rep/FetalHASTEReposatoryFromBCH_Anon/')
NO_LABEL = -1

class_weight = [1.0, 1.0, 1.0]
random_idx = []


class HASTEDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=False, include_unlabel=False, use_mask=False, transform=None, num_data=0, is_test=False):
        self.is_train = is_train
        self.use_mask = use_mask
        if is_train:
            self.data = read_labeled_data(['train_all.npy'])
            
            if len(random_idx) == 0:
                random_idx.extend(list(range(len(self.data))))
                random.shuffle(random_idx)
            
            idx_o, idx_g, idx_b = [], [], []
            if num_data > 0:
                self.data = list(self.data[idx] for idx in random_idx[:num_data])
            num_o, num_g, num_b = 0, 0, 0
            for i, d in enumerate(self.data):
                if d[0] == 0:
                    num_o += 1
                elif d[0] == 1:
                    num_g += 1
                else:
                    num_b += 1
            print('number of labeled data: %d, %d, %d, %d' % (len(self.data), num_o, num_g, num_b))
            class_weight[0], class_weight[1], class_weight[2] = num_g/num_o, 1.0, num_g/num_b
            print('class weight:', class_weight)
            self.labeled_idxs = range(len(self.data))
            if include_unlabel:
                ud = read_unlabeled_data()
                self.unlabeled_idxs = range(len(self.data), len(self.data) + len(ud))
                self.data = self.data + ud
            else:
                self.unlabeled_idxs = []
        else:
            self.data = read_labeled_data(['test_all.npy'] if is_test else ['val_all.npy'])
            self.labeled_idxs = range(len(self.data))
            self.unlabeled_idxs = [] 
            
            num_o, num_g, num_b = 0, 0, 0
            for i, d in enumerate(self.data):
                if d[0] == 0:
                    num_o += 1
                elif d[0] == 1:
                    num_g += 1
                else:
                    num_b += 1
            print('class weight (val):', [num_g/num_o, 1.0, num_g/num_b])
            
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        target, (cx, cy, r, std), filename = self.data[idx]
        img = read_dicom(filename)
        if self.is_train and self.use_mask:
            if r:
                mask = np.zeros_like(img)
                cv2.circle(mask, (cx, cy), r + np.random.randint(std, 2*std+1), 1, thickness=-1)
                masked = img * mask
            else:
                masked = img.copy()
            img = (img, masked)
            
        if self.transform is not None:
            img = self.transform(img)
        return img, target
        
        
def read_labeled_data(names):
    with open(os.path.join(obj_dir, "labeled.obj"),"rb") as filehandle:
        mask = pickle.load(filehandle)
    path = PATH_LABEL
    path_data = PATH_LABELED_DATA
    data = []
    for d in np.concatenate([np.load(os.path.join(path, name)).astype(np.str_) for name in names]):
        if d[0] == '1':
            c = int(d[1]) + 1
        elif d[0] == '0':
            c = 0
        else:
            raise Exception('unknown roi label')
        p = os.path.join(path_data, d[2], d[3], 'dicoms', d[4]+'.dcm')
        data.append((c, mask[(d[2], d[3])], p))
    data = sorted(data, key=lambda x:x[-1])
    return data
    
    
def read_unlabeled_data():
    with open(os.path.join(obj_dir, "unlabeled.obj"),"rb") as filehandle:
        mask = pickle.load(filehandle)
    data = []
    f0 = PATH_UNLABELED_DATA
    for f1 in os.listdir(f0):
        for f2 in os.listdir(os.path.join(f0, f1)):
            for f3 in os.listdir(os.path.join(f0, f1, f2)):
                if f3.startswith('T2_HASTE'):
                    flag = True
                    sn = []
                    dd = []
                    for filename in os.listdir(os.path.join(f0, f1, f2, f3)):
                        f = os.path.join(f0, f1, f2, f3, filename)
                        if filename.endswith('.dcm') and filename[4] == '-':
                            sn.append(int(filename[:4]))
                            if 50 * 1024 <= os.path.getsize(f) <= 400 * 1024:
                                dd.append(f)
                            else:
                                flag = False
                                break
                    if flag:
                        mask_res = mask[f1, f2, f3]
                        data.extend([(NO_LABEL, mask_res, fn) for fn in dd])
                                
    print("unlabeled data: %d" % len(data))
    return data
    

def read_dicom(filename):
    dcm = pydicom.dcmread(filename)
    img = dcm.pixel_array.astype(np.float32)
    assert img.ndim == 2
    return img
