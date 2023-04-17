from scipy import io
import os
import mne
import numpy as np
from tqdm import tqdm
import random
import pdb

def load_from_single_path(file_path):
    # load the data, get peak time
    loaded = io.loadmat(file_path)
    peak_time = [loaded[list(loaded.keys())[3]][0][i][1][0][0] for i in range(loaded[list(loaded.keys())[3]].shape[1])] # second index -> # of peaks
    peak_time_ms = [int(t * 1000) for t in peak_time]
    data_MEG = loaded['data_raw']
    data_MEG_filtered = mne.filter.filter_data(data_MEG, 1000, 1, 100)
    # loop through peak time
    positive_samples = []
    negative_samples = []
    for p in tqdm(peak_time_ms):
        # identify region
        region = None
        max_ind = 0
        max_num = 0
        for i in range(306):
            if abs(data_MEG_filtered[i][p]) > max_num:
                max_num = abs(data_MEG_filtered[i][p])
                max_ind = i
        region = channel_grouping_inv[channel_name_dict[max_ind]]
        for i in np.arange(p-220, p-80, 5):
            curr_data = data_MEG_filtered[channel_grouping_index[region], i:i+300]
            if curr_data.shape[1] != 300: 
                    continue
            if region == "LO" or region == "RO":
                new_row = np.zeros((3, 300))
                padded_data = np.vstack((curr_data, new_row))
                positive_samples.append(padded_data)
            else: 
                positive_samples.append(curr_data)
    while len(negative_samples) < len(positive_samples):
        random_group = random.choice(list(channel_grouping_index.keys()))
        random_num = random.randint(0, data_MEG_filtered.shape[1] - 300)
        for p in peak_time_ms:
            if p < random_num or p > random_num + 300:
                curr_data = data_MEG_filtered[channel_grouping_index[random_group], random_num:random_num+300]
                if curr_data.shape[1] != 300: 
                    continue
                if random_group == "LO" or random_group == "RO":
                    new_row = np.zeros((3, 300))
                    padded_data = np.vstack((curr_data, new_row))
                    negative_samples.append(padded_data)
                else:
                    negative_samples.append(curr_data)
    returned_data = positive_samples + negative_samples
    returned_label = [1] * len(positive_samples) + [0] * len(negative_samples)
    return (returned_data, returned_label)

if __name__ == "__main__":

    # create channel grouping
    with open('channel_name.txt', 'r') as f:
        content = f.readline()
    content = content.split('	')
    content = [s.strip('\'').strip('MEG') for s in content]
    channel_name_pair = [(i, content[i]) for i in range(len(content))]
    channel_name_pair_inv = [(content[i], i) for i in range(len(content))]
    # create two dictionary: channel_name_dict: channel index -> channel name, channel_name_dict_inv: channel name -> channel index
    channel_name_dict = dict(channel_name_pair)
    channel_name_inv_dict = dict(channel_name_pair_inv)
    # divide channels into 8 groups based on brain anatonic structure. # labeled manually
    channel_grouping = {
        "LF": ['0121', '0122', '0123', '0311', '0312', '0313', '0341', '0342', '0343', '0321', '0322', '0323', '0511', '0512', '0513', '0541', '0542', '0543', '0331', '0332', '0333', '0521', '0522', '0523', '0531', '0532', '0533', '0611', '0612', '0613', '0641', '0642', '0643', '0821', '0822', '0823', '0621', '0622', '0623'],
        "RF": ['0811', '0812', '0813', '1011', '1012', '1013', '0911', '0912', '0913', '0941', '0942', '0943', '1021', '1022', '1023', '1031', '1032', '1033', '0921', '0922', '0923', '0931', '0932', '0933', '1241', '1242', '1243', '1211', '1212', '1213', '1231', '1232', '1233', '1221', '1222', '1223', '1411', '1412', '1413'],
        "LT": ['0111', '0112', '0113', '0131', '0132', '0133', '0211', '0212', '0213', '0221', '0222', '0223', '0141', '0142', '0143', '1511', '1512', '1513', '0241', '0242', '0243', '0231', '0232', '0233', '1541', '1542', '1543', '1521', '1522', '1523', '1611', '1612', '1613', '1621', '1622', '1623', '1531', '1532', '1533'],
        "LP": ['0411', '0412', '0413', '0421', '0422', '0423', '0631', '0632', '0633', '0441', '0442', '0443', '0431', '0432', '0433', '0711', '0712', '0713', '1811', '1812', '1813', '1821', '1822', '1823', '0741', '0742', '0743', '1631', '1632', '1633', '1841', '1842', '1843', '1831', '1832', '1833', '2011', '2012', '2013'],
        "RP": ['1041', '1042', '1043', '1111', '1112', '1113', '1121', '1122', '1123', '0721', '0722', '0723', '1141', '1142', '1143', '1131', '1132', '1133', '0731', '0732', '0733', '2211', '2212', '2213', '2221', '2222', '2223', '2241', '2242', '2243', '2231', '2232', '2233', '2441', '2442', '2443', '2021', '2022', '2023'], 
        "RT": ['1311', '1312', '1313', '1321', '1322', '1323', '1441', '1442', '1443', '1421', '1422', '1423', '1431', '1432', '1433', '1341', '1342', '1343', '1331', '1332', '1333', '2611', '2612', '2613', '2411', '2412', '2413', '2421', '2422', '2423', '2641', '2642', '2643', '2621', '2622', '2623', '2631', '2632', '2633'],
        "LO": ['1641', '1642', '1643', '1721', '1722', '1723', '1711', '1712', '1713', '1731', '1732', '1733', '1941', '1942', '1943', '1741', '1742', '1743', '1911', '1912', '1913', '1921', '1922', '1923', '1931', '1932', '1933', '2141', '2142', '2143', '2041', '2042', '2043', '2111', '2112', '2113'],
        "RO": ['2121', '2122', '2123', '2031', '2032', '2033', '2131', '2132', '2133', '2341', '2342', '2343', '2331', '2332', '2333', '2311', '2312', '2313', '2321', '2322', '2323', '2541', '2542', '2543', '2511', '2512', '2513', '2431', '2432', '2433', '2521', '2522', '2523', '2531', '2532', '2533']    
    }

    channel_grouping_inv = {}
    for key in channel_grouping.keys():
        curr_lst = channel_grouping[key]
        for item in curr_lst:
            channel_grouping_inv[item] = key

    # create groups based on index from 0 to 305
    channel_grouping_index = {}
    for key in channel_grouping.keys():
        channel_grouping_index[key] = []
        list_of_name = channel_grouping[key]
        for name in list_of_name:
            channel_grouping_index[key].append(channel_name_inv_dict[name])

    # dataset directory
    directory = "./dataset"

    # Check if the directory exists
    if os.path.exists(directory):
        print("dataset directory already exists!")
    
    else:
        print("creating dataset ... ")
        data_folder_path = './raw_data'
        data_paths_list = []

        for single_folder_name in os.listdir(data_folder_path):
            single_folder_path = os.path.join(data_folder_path, single_folder_name)
            for single_file_name in os.listdir(single_folder_path):
                data_paths_list.append(os.path.join(single_folder_path, single_file_name))

        training_data = []
        training_label = []
        val_data = []
        val_label = []
        test_data = []
        test_label = []
        for i in range(2, len(data_paths_list)):
            curr_path = data_paths_list[i]
            curr_data, curr_label = load_from_single_path(curr_path)
            training_data += curr_data
            training_label += curr_label
        val_data, val_label = load_from_single_path(data_paths_list[0])
        test_data, test_label = load_from_single_path(data_paths_list[1])

        # Check if the directory exists, and create it if it doesn't
        os.mkdir(directory)

        # Create the train_small, label_small txt file in the directory
        file_path_train = os.path.join(directory, "train_data.npy")
        file_path_train_label = os.path.join(directory, "train_label.npy")
        file_path_val = os.path.join(directory, "val_data.npy")
        file_path_val_label = os.path.join(directory, "val_label.npy")
        file_path_test = os.path.join(directory, "test_data.npy")
        file_path_test_label = os.path.join(directory, "test_label.npy")
        
        #pdb.set_trace()

        np.save(file_path_train, training_data)
        np.save(file_path_train_label, training_label)
        np.save(file_path_val, val_data)
        np.save(file_path_val_label, val_label)
        np.save(file_path_test, test_data)
        np.save(file_path_test_label, test_label)

        '''
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
        '''

        print("dataset created!")
    


