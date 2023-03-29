import os
import numpy as np
from torch.utils import data

num_classes = 21
root = './dataset'

def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_path = os.path.join(root, 'train_small.npy')
        label_path = os.path.join(root, 'label_small.npy')
        trainset = np.load(train_path) 
        labelset = np.load(label_path)
        for x, label in zip(trainset, labelset):
            items.append((x, label))
    elif mode == 'val':
        print("TODO")
    else:
        print("TODO")
    return items


class MEG(data.Dataset):
    def __init__(self, mode, transform=None, target_transform=None):
        self.data_pairs = make_dataset(mode)
        if len(self.data_pairs) == 0:
            raise RuntimeError('Found 0 data, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        input_data = self.data_pairs[index][0]
        target_label = self.data_pairs[index][1]
        if self.transform is not None:
            input_data = self.transform(input_data)
        if self.target_transform is not None:
            target_label = self.target_transform(target_label)
        return input_data, target_label

    def __len__(self):
        return len(self.data_pairs)