from EMSnet import *
from MEG import MEG
from plot import plots
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
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

def save_model(model):
    model_directory = "./saved_model"
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)
    filename = 'best_model.pkl'
    model_path = os.path.join(model_directory, filename)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    file.close()

epochs = 10
n_class = 2
batch_size = 64
best_val_loss = 100

target_transform = MaskToTensor()
input_transform = standard_transforms.Compose([
        Standardization(),
        MaskToTensor()
    ])

train_dataset = MEG('train', transform=input_transform, target_transform=target_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)

val_dataset = MEG('val', transform=input_transform, target_transform=target_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size = 1, shuffle=True)

test_dataset = MEG('test', transform=input_transform, target_transform=target_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size = 1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EMS_Nets(batch_size)
model.apply(init_weights)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())
scheduler = CosineAnnealingLR(optimizer, T_max=5)
criterion = torch.nn.BCELoss()

def acc(pred, target):
    the_same = (pred == target)
    total_number = torch.sum(torch.ones_like(target))
    return torch.sum(the_same) / total_number

def F1_metrics(y_true, y_pred):
    # Calculate TP, TN, FP, FN
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate precision, recall, and F1-score
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    # Return the metrics
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

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
    train_acc_list = []
    val_acc_list = []
    val_loss_list = []
    early_stop_epoch = 0
    min_val_loss = None

    for epoch in tqdm(range(epochs)):
        model.train()
        ts = time.time()
        # loss_epoch = []
        train_acc = []
        for iter, (inputs, labels) in enumerate(train_loader):
            
            # reset optimizer gradients
            optimizer.zero_grad()
            
            # adjust labels size to be batch_size * 1
            labels = torch.unsqueeze(labels, dim = 1)
            
            inputs = inputs.to(device) # transfer the input to the same device as the model's
            labels = labels.to(device) # transfer the labels to the same device as the model's
            
            #pdb.set_trace()
            outputs = model(inputs)
            pred = (outputs >= 0.5)
            train_acc.append(acc(pred, labels))
            #pdb.set_trace()

            # calculate loss
            loss = criterion(outputs, labels)

            # backpropagation
            loss.backward()
            loss_epoch.append(loss.item())

            # update the weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        scheduler.step()

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        print("Train Accuracy: ", np.mean(train_acc))
        train_acc_list.append(np.mean(train_acc))
        val_loss, val_acc = val(epoch)

        if min_val_loss is None or min_val_loss > val_loss:
            min_val_loss = val_loss
            early_stop_epoch = epoch

        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        train_loss_list.append(np.mean(loss_epoch))

    plots(train_acc_list, val_acc_list, early_stop_epoch, "EMS-Nets", "Acc")
    plots(train_loss_list, val_loss_list, early_stop_epoch, "EMS-Nets", "Loss")

def val(epoch):
    global best_val_loss
    model.eval() # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    accuracy = []
    TP_tot, TN_tot, FP_tot, FN_tot = 0, 0, 0, 0

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):

            # adjust labels size to be batch_size * 1
            labels = torch.unsqueeze(labels, dim = 1)
            
            inputs = inputs.to(device) # transfer the input to the same device as the model's
            labels = labels.to(device) # transfer the labels to the same device as the model's
            
            outputs = model(inputs)
            pred = (outputs >= 0.5) # pred shape N * H * W
            
            val_loss = criterion(outputs, labels)
            val_acc = acc(pred, labels)
            #pdb.set_trace()
            TP_tot += (labels[0][0] == 1) & (pred[0][0] == 1)
            TN_tot += (labels[0][0] == 0) & (pred[0][0] == 0)
            FP_tot += (labels[0][0] == 0) & (pred[0][0] == 1)
            FN_tot += (labels[0][0] == 1) & (pred[0][0] == 0)

            losses.append(val_loss.item())
            accuracy.append(val_acc.item())
    
    precision = TP_tot / (TP_tot + FP_tot)
    recall = TP_tot / (TP_tot + FN_tot)

    print(f"Val Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"Val Acc at epoch: {epoch} is {np.mean(accuracy)}")
    print(f"Val: TP: {TP_tot}, TN: {TN_tot}, FP: {FP_tot}, FN: {FN_tot}")
    print(f"Val: Precision: {precision}, Recall: {recall}, F-1 score: {2 * precision * recall / (precision + recall)}")

    model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    if np.mean(losses) < best_val_loss:
        save_model(model)
        best_val_loss = np.mean(losses)
    
    return np.mean(losses), np.mean(accuracy)

def test():
    model_directory = "./saved_model"
    filename = 'best_model.pkl'
    model_path = os.path.join(model_directory, filename)
    with open(model_path, 'rb') as file:
        curr_model = pickle.load(file)
    file.close()

    curr_model.eval() # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    accuracy = []
    TP_tot, TN_tot, FP_tot, FN_tot = 0, 0, 0, 0

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in tqdm(enumerate(test_loader)):

            # adjust labels size to be batch_size * 1
            labels = torch.unsqueeze(labels, dim = 1)
            
            inputs = inputs.to(device) # transfer the input to the same device as the model's
            labels = labels.to(device) # transfer the labels to the same device as the model's
            
            outputs = curr_model(inputs)
            pred = (outputs >= 0.5) # pred shape N * H * W
            
            val_loss = criterion(outputs, labels)
            val_acc = acc(pred, labels)
            #pdb.set_trace()
            TP_tot += (labels[0][0] == 1) & (pred[0][0] == 1)
            TN_tot += (labels[0][0] == 0) & (pred[0][0] == 0)
            FP_tot += (labels[0][0] == 0) & (pred[0][0] == 1)
            FN_tot += (labels[0][0] == 1) & (pred[0][0] == 0)

            losses.append(val_loss.item())
            accuracy.append(val_acc.item())
    
    precision = TP_tot / (TP_tot + FP_tot)
    recall = TP_tot / (TP_tot + FN_tot)

    print(f"(Sliding window data) Testing Loss is {np.mean(losses)}")
    print(f"(Sliding window data) Testing Acc is {np.mean(accuracy)}")
    print(f"(Sliding window data) Testing: TP: {TP_tot}, TN: {TN_tot}, FP: {FP_tot}, FN: {FN_tot}")
    print(f"(Sliding window data) Testing: Precision: {precision}, Recall: {recall}, F-1 score: {2 * precision * recall / (precision + recall)}")

    return np.mean(losses), np.mean(accuracy)


if __name__ == "__main__":
    
    train()
    test()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()


