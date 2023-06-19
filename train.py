import argparse
import json
import logging
import sys
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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
        torch.nn.init.normal_(m.bias.data)

# def save_model(model):
#     model_directory = "./saved_model"
#     if not os.path.exists(model_directory):
#         os.mkdir(model_directory)
#     filename = 'best_model.pkl'
#     model_path = os.path.join(model_directory, filename)
#     with open(model_path, 'wb') as file:
#         pickle.dump(model, file)
#     file.close()

n_class = 2
best_model = None
criterion = torch.nn.BCELoss()
target_transform = MaskToTensor()
input_transform = standard_transforms.Compose([
        Standardization(),
        MaskToTensor()
    ])

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

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def train(args):
        
    min_val_loss = None

    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    model = EMS_Nets(args.batch_size)
    model.apply(init_weights)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    train_dataset = MEG('train', transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle=True)

    val_dataset = MEG('val', transform=input_transform, target_transform=target_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size = args.test_batch_size, shuffle=False)

    test_dataset = MEG('test', transform=input_transform, target_transform=target_transform)
    test_loader = DataLoader(dataset=test_dataset, barch_size = args.test_batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    scheduler = CosineAnnealingLR(optimizer, T_max=5)

    loss_epoch = []
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_loss_list = []
    early_stop_epoch = 0
    min_val_loss = None

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        ts = time.time()
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
            curr_acc = acc(pred, labels)
            train_acc.append(curr_acc)
            #pdb.set_trace()

            # calculate loss
            loss = criterion(outputs, labels)

            # backpropagation
            loss.backward()
            loss_epoch.append(loss.item())

            # update the weights
            optimizer.step()

            if iter % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [({:.2f}%)] Loss: {:.6f}".format(
                        epoch,
                        100.0 * iter / len(train_loader),
                        loss.item(),
                    )
                )
        scheduler.step()

        logger.info(
                    "Finish training Epoch: {}, Accuracy: {:.2f}%".format(
                        epoch,
                        100 * np.mean(train_acc),
                    )
                )
        train_acc_list.append(np.mean(train_acc))
        val_loss, val_acc = val(model, val_loader, device)

        if best_val_loss > val_loss:
            best_model = model
            best_val_loss = val_loss

        if min_val_loss is None or min_val_loss > val_loss:
            min_val_loss = val_loss
            early_stop_epoch = epoch
            best_model = model

        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        train_loss_list.append(np.mean(loss_epoch))

    save_model(model, args.model_dir)

    plots(train_acc_list, val_acc_list, early_stop_epoch, "EMS-Nets", "Acc")
    plots(train_loss_list, val_loss_list, early_stop_epoch, "EMS-Nets", "Loss")

def val(model, val_loader, device):
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

    logger.info(
        "Validation set: Average loss: {:.4f}, Accuracy: ({:.2f}%), TP: {}, TN: {}, FP: {}, FN: {}, Precision: {}, Recall: {}, F-1 score: {:.2f}%\n".format(
            np.mean(losses), 100.0 * np.mean(accuracy), TP_tot, TN_tot, FP_tot, FN_tot, precision, recall, 100.0 * 2 * precision * recall / (precision + recall)
        )
    )

    # if np.mean(losses) < best_val_loss:
    #     save_model(model)
    #     best_val_loss = np.mean(losses)
    
    return np.mean(losses), np.mean(accuracy)

'''
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
'''

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, metavar="LR", help="learning rate (default: 0.0002)"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, metavar="BETA1", help="beta1 (default: 0.9)"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, metavar="BETA2", help="beta2 (default: 0.999)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
    
    #test()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()


