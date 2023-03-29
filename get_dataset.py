from scipy import io
import os
import mne
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    directory = "./dataset"

    # Check if the directory exists
    if os.path.exists(directory):
        print("dataset directory already exists!")
    
    else:
        print("creating dataset ... ")
        loaded = io.loadmat('./raw_data/case_2225/case_2225_with_spike_dipoles_sleep_2.mat')
        peak_time = [loaded['Dipole_sleep_2'][0][i][1][0][0] for i in range(loaded['Dipole_sleep_2'].shape[1])] # second index -> # of peaks
        data_MEG = loaded['data_raw']
        data_MEG_filtered = mne.filter.filter_data(data_MEG, 1000, 1, 100)
        first = data_MEG_filtered[:39] # choose the first 39 channels as sample input

        data_MEG_processed = []
        label = []

        for i in tqdm(range(0, 509000, 300)):
            if i == 508800:
                break
            data_MEG_processed.append(first[:, i:i+300])

            l, r = i, i + 300
            true_label = 0
            for t in peak_time:
                if l <= t*1000 and t*1000 < r:
                    label.append(1)
                    true_label = 1
                    break
            if true_label == 0:
                label.append(0)
            
        file_name_train = "train_small.npy"
        file_name_label = "label_small.npy"

        # Check if the directory exists, and create it if it doesn't
        os.mkdir(directory)

        # Create the train_small, label_small txt file in the directory
        file_path_train = os.path.join(directory, file_name_train)
        file_path_label = os.path.join(directory, file_name_label)
        np.save(file_path_train, data_MEG_processed)
        np.save(file_path_label, label)

        print("dataset created!")
    


