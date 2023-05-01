import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def plots(trainEpoch, valEpoch, earlyStop, model, type):
    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. earlyStop=-1 means the last epoch was the best one
    """
    saveLocation="./results/plot/"
    fig1, ax1 = plt.subplots(figsize=((12, 12)))
    epochs = np.arange(1,len(trainEpoch)+1,1)
    if type == "Acc":
        ax1.plot(epochs, trainEpoch, 'r', label="Training Accuracy")
        ax1.plot(epochs, valEpoch, 'g', label="Validation Accuracy")
    if type == "Loss":
        ax1.plot(epochs, trainEpoch, 'r', label="Training Loss")
        ax1.plot(epochs, valEpoch, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpoch[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=30 )
    plt.yticks(fontsize=30)
    if type == "Acc":
        ax1.set_title('Accuracy Plots', fontsize=30.0)
        ax1.set_xlabel('Epochs', fontsize=30.0)
        ax1.set_ylabel('Accuracy', fontsize=30.0)
        ax1.legend(loc="upper right", fontsize=30.0)
        plt.savefig(saveLocation+model+"_acc.eps")
    elif type == "Loss":
        ax1.set_title('Loss Plots', fontsize=30.0)
        ax1.set_xlabel('Epochs', fontsize=30.0)
        ax1.set_ylabel('Loss', fontsize=30.0)
        ax1.legend(loc="upper right", fontsize=30.0)
        plt.savefig(saveLocation+model+"_loss.eps")
    plt.show()
