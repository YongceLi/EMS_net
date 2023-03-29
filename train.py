from EMSnet import *
from MEG import MEG
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import torch
import time
import gc
import numpy as np
import os
from tqdm import tqdm
from operator import add
import pandas as pd
import pickle
import pdb

class MaskToTensor(object):
    """ change the object to Pytorch tensor """
    def __call__(self, data):
        return torch.from_numpy(np.array(data, dtype=np.float32))

class Standardization(object):
    """ standardize the data """
    def __call__(self, data):
        return (data - np.mean(data)) / np.std(data)

def init_weights(m):
    """ initializa weights to the given model (xavier uniform) """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases

epochs = 20
n_class = 2
batch_size = 4

target_transform = MaskToTensor()
input_transform = standard_transforms.Compose([
        Standardization(),
        MaskToTensor()
    ])

train_dataset = MEG('train', transform=input_transform, target_transform=target_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EMS_Nets(batch_size)
model.apply(init_weights)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCELoss()

def train():
    print('train start')
    if not os.path.exists('./results'):
        os.mkdir('./results')
    
    '''
    best_val_loss = np.inf
    best_epoch = 0
    train_loss_list = []
    val_loss_list = []
    '''
    loss_epoch = []
    train_loss_list = []

    for epoch in tqdm(range(epochs)):
        model.train()
        ts = time.time()
        # loss_epoch = []
        for iter, (inputs, labels) in enumerate(train_loader):
            
            # reset optimizer gradients
            optimizer.zero_grad()
            
            # adjust labels size to be batch_size * 1
            labels = torch.unsqueeze(labels, dim = 1)
            
            inputs = inputs.to(device) # transfer the input to the same device as the model's
            labels = labels.to(device) # transfer the labels to the same device as the model's
            
            #pdb.set_trace()
            outputs = model(inputs)
            
            # calculate loss
            loss = criterion(outputs, labels) 

            # backpropagation
            loss.backward()
            loss_epoch.append(loss.item())

            # update the weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        train_loss_list.append(np.mean(loss_epoch))
    
if __name__ == "__main__":
    
    train()
    # test()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()