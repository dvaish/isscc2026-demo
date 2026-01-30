# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
from scipy import signal


# %%
def get_feature(data):
    nch, ndata = data.shape
    filter = signal.butter(2, 1, btype='high', output='sos', fs=1000)
    data = signal.sosfilt(filter, data, axis=-1, zi=None)
    data_r = data.reshape(nch, ndata//50, 50)
    means = np.mean(data_r, axis=-1, keepdims=True)
    return np.sum(abs(data_r-means), axis=-1)/50

# %%
def load_dataset(filename:str, splits:int=8, raw=False):
    dataset = np.load(filename)

    # Fix for variable length arrays in npz files
    if "npz" in filename:
        lengths = list(map(lambda x: dataset[x].shape[-1], dataset.keys()))
        length = min(lengths)
        dataset = np.array([dataset[key][:, :length] for key in dataset.keys()])

    data_lst = []
    for i in range(11):
        start = 30000 + i*11000 + 3500
        stop = start + 4000
        lst = []
        for j in range(splits):
            lst.append(dataset[j, :(64 // splits), start:stop])
        data = np.concatenate(lst, axis=0)
        data_lst.append(data)
    if raw:
        return np.concatenate(data_lst, axis=-1)
    else:
        return get_feature(np.concatenate(data_lst, axis=-1))

def build_dataset(filename='PlaybackOld/adc_raw_{trial}_{setting}.npy', splits=8, raw=False):
    trial_lst = []
    label_lst = []
    for trial in range(5):
        feat_lst = []
        for setting in [3, 2, 1, 0]:
            feat_lst.append(load_dataset(filename.format(trial=trial, setting=setting), splits=splits, raw=raw))
        
        labels = np.array(sum([[i]*80 for i in range(11)], []))
        trial_lst.append(np.stack(feat_lst))
        label_lst.append(labels)
    return trial_lst, label_lst

def cut(dataset, raw=False):
    data_lst = []
    for i in range(11):
        start = 30000 + i*11000 + 3500
        stop = start + 4000
        # start = 41000 + i*11000 + 1000
        # stop = start + 11000 - 1000
        lst = []
        for j in range(splits):
            lst.append(dataset[j, :(64 // splits), start:stop])
        data = np.concatenate(lst, axis=0)
        data_lst.append(data)
    if raw:
        return np.concatenate(data_lst, axis=-1)
    else:
        return get_feature(np.concatenate(data_lst, axis=-1))

# %%
from typing import Union, Iterable
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def select_noise_floors(data: list[np.ndarray],
                        floor_idx: Union[list[int], int]) -> np.ndarray:
    """ Select the correct noise floor indices. 
    Return a num_trials list of (nch x npts) arrays
    Each element in the list corresponds to a specific trial
    """
    if isinstance(floor_idx, Iterable):
        # Overall data list is a list[list[array[nch x npts]]]
        return [np.stack([d[n,ch,:] for ch, n in enumerate(floor_idx)]) for d in data]
    else:
        return [d[floor_idx, :, :] for d in data]


def build_log_reg_model(x_train: np.ndarray, y_train: np.ndarray, C=10):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_s = scaler.transform(x_train)
    model = LogisticRegression(
            l1_ratio=0, C=C, max_iter=int(1e5),
            class_weight='balanced')
    model.fit(x_train_s, y_train)
    return scaler, model


def get_log_reg_weights(model: LogisticRegression,
                        x_train: np.ndarray, 
                        y_train: np.ndarray) -> np.ndarray:
    coef = model.coef_
    if len(coef.shape) > 1:
        coef_reshaped = coef.reshape(coef.shape[0], x_train.shape[-1], -1)
        return np.max(np.average(abs(coef_reshaped), axis=0), axis=-1)
    else:
        return abs(coef)


def get_test_accuracy(x_test, y_test, model, scaler):
    x_test_s = scaler.transform(x_test)
    y_pred = model.predict(x_test_s)
    accuracy = sum(y_pred == y_test) / len(y_pred) * 100
    return accuracy


def get_idcs(coefs, noise_floor_min, noise_floor_max):
    importance = abs(coefs)
    min_importance = min(importance[np.nonzero(importance)])
    slope = (np.log10(noise_floor_min) - np.log10(noise_floor_max)) / (max(importance) - min_importance)
    noise_floors = 10**((importance - min_importance) * slope + np.log10(noise_floor_max))
    n = np.array(_noise_floors)
    return [np.argmin(abs(n-x)) for x in noise_floors]


def get_ch_ranking(model):
    coef = [(i, abs(c)) for i, c in enumerate(model.coef_[0])]
    coef.sort(key=lambda x: x[1])
    coef.reverse()
    return np.array([c[0] for c in coef])


def run_enob_sweep(trials_data, trials_labels, noise_floors, C=10):
    """ 
    Run the ENOB Sweep
    """
    acc_arr, weights_arr = [], []

    num_floors = len(noise_floors)
    for idx in tqdm(range(num_floors)):
        acc, weights = [], []
        trials_data_s = select_noise_floors(trials_data, idx)
        for i in range(5):
            x_train = np.concatenate(
                    trials_data_s[:i]+trials_data_s[i+1:], axis=-1).T
            y_train = np.concatenate(trials_labels[:i]+trials_labels[i+1:])
            # print(f"Building Model {i+1} of 5")
            scaler, model = build_log_reg_model(x_train, y_train, C=C)
            x_test = trials_data_s[i].T
            y_test = trials_labels[i]  
            # print(f"Testing Model {i+1} of 5")
            acc.append(get_test_accuracy(x_test, y_test, model, scaler))
            weights.append(get_log_reg_weights(model, x_train, y_train))
        acc_arr.append(acc)
        weights_arr.append(weights)
    return acc_arr, weights_arr


# %%
def run_sparse_model(trials_data, trials_labels, noise_floors, noise_idx, C=10) -> None:
    trials_data = select_noise_floors(trials_data, noise_idx)
    nch, _ = trials_data[0].shape
    acc_arr, weights_arr = [], []

    for i in range(5):
        print(f"Running on trial {i+1} of 5")
        y_train = np.concatenate(trials_labels[:i]+trials_labels[i+1:])
        y_test = trials_labels[i]
        x_train = np.concatenate(trials_data[:i]+trials_data[i+1:], axis=-1).T
        x_test = trials_data[i].T
        ch_selection = np.ones(nch)
        acc, weights = [], []
        for ch in range(nch, 0, -1):
            print(f"Running with {ch} channels")
            sel = np.array([idx for idx in range(nch) if ch_selection[idx]>0])
            scaler, model = build_log_reg_model(x_train[:, sel], y_train, C=C)
            acc.append(get_test_accuracy(x_test[:, sel], y_test, model, scaler))
            curr_weights = get_log_reg_weights(model, x_train[:, sel], y_train)
            overall_weights = np.zeros(nch)
            overall_weights[sel] = curr_weights
            weights.append(overall_weights)
            min_weight_idx = min(
                    zip(sel, curr_weights), key=lambda x: abs(x[1]))[0]
            ch_selection[min_weight_idx] = 0
        acc_arr.append(acc)
        weights_arr.append(weights)
    return acc_arr, weights_arr

    # %%
def run_adaptive_model(trials_data, trials_labels, enabled_settings=[0, 1, 2, 3], train_setting=None, train_noise=0, selection=0, C=10, sim_weights=None, sim_settings=None, mode='linear'):
    enabled_settings = np.array(enabled_settings)
    if train_setting is None:
        train_setting = min(enabled_settings)
    trials_data_s = select_noise_floors(trials_data, train_setting)
    nch, _ = trials_data_s[0].shape
    trials_labels_s = [l[0] for l in trials_labels]

    acc_arr, weights_arr, settings_arr = [], [], []
    for i in range(5):
        # print(f"Running on trial {i} of 5")
        y_train = np.concatenate(trials_labels[:i]+trials_labels[i+1:])
        y_test = trials_labels[i]
        x_train = np.concatenate(trials_data_s[:i]+trials_data_s[i+1:], axis=-1).T
        x_train = x_train + np.random.normal(0, train_noise, x_train.shape)
        scaler, model = build_log_reg_model(x_train, y_train, C=C)
        

        if sim_weights is not None:
            curr_weights = sim_weights[i]
        else:
            num_settings = len(enabled_settings)
            curr_weights = get_log_reg_weights(model, x_train, y_train)
            print(curr_weights)
            if mode == 'exponential':
                curr_weights = np.exp(curr_weights) / np.sum(np.exp(curr_weights)) # Implement a softmax distribution
            elif mode == 'quadratic':
                curr_weights = curr_weights**2
            bin_size = (max(curr_weights)-min(curr_weights))/num_settings
            weights = (curr_weights-min(curr_weights))/bin_size
            thresholds = np.linspace(-0.5, num_settings+0.5, num_settings+2)
            
        if sim_settings is not None:
            settings = sim_settings[i]
        else:
            # Calculatons of setting thresholds
            
            # thresholds = np.array([-0.5]+list(
            #         np.arange(0.5, num_settings, 1))+[num_settings+0.5])
            # fig, ax = plt.subplots(2) # sharex=True)
            # ax[0].plot(curr_weights)
            # ax[0].plot(weights)
            
            # For now, just set the lowest weights to 0, don't remap the settings.
            settings = np.zeros(weights.shape).astype(int)
            for idx in range(num_settings+1):
                arg = (weights<=thresholds[idx+1])&(weights>=thresholds[idx])
                settings = np.where(arg, idx, settings)
            # Settings goes from 0 to n where n represents "best" setting
            # and 0 represents worst setting
            # In the real array, noise index 0 corresponds to best, and n to worst. 
            # ax[1].plot(settings)

            settings = (num_settings - 1) - settings
            settings = enabled_settings[settings]
            # plt.hist(settings, bins=range(0, num_settings+1), alpha=0.5)


        # do the selection
        # for now, just set the lowest weights to 0, don't remap the settings.
        sorted_weight_idcs = np.argsort(weights)
        dropped_idcs = sorted_weight_idcs[:selection]
            
        # ax[0].plot((3-settings)/3*max(curr_weights))
        settings_p = settings.copy()
        # settings_p[np.argwhere(settings == 2)] = 2

        trials_data_sel = select_noise_floors(trials_data, settings_p)
        x_retrain = np.concatenate(
                trials_data_sel[:i]+trials_data_sel[i+1:], axis=-1).T
        x_retrain[:, dropped_idcs] = 0
        scaler, model = build_log_reg_model(x_retrain, y_train, C=C)
        # ax[0].plot(get_log_reg_weights(model, x_train, y_train))
        x_test = trials_data_sel[i].T
        x_test[:, dropped_idcs] = 0
        acc_arr.append(get_test_accuracy(x_test, y_test, model, scaler))
        weights_arr.append(weights)
        settings[dropped_idcs] = -1
        settings_arr.append(settings)
        # ax[1].hist(settings, bins=range(-1, 5))
        
    return acc_arr, weights_arr, settings_arr

settings_to_pow = np.array([2.21E-06, 6.40E-07, 2.22E-07, 1.50E-07]) * 0.8

if __name__ == "__main__":

    # %%
    splits = 4
    trial_lst, label_lst = build_dataset(filename='Playback/emg/user1/adc_raw_{trial}_21_{setting}.npz', splits=4, raw=False)
    trial_lst = np.array(trial_lst)

    data = np.array(trial_lst)
    data_white = (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)

    # %%
    

    # %%
    base_acc_arr, base_weights_arr = run_enob_sweep(trial_lst, label_lst, [0, 1, 2, 3], C=10)

    # %%
    # sparse_accs = []
    # sparse_weights = []
    # for i in range(4):
    #     sparse_acc_arr, sparse_weights_arr = run_sparse_model(trial_lst, label_lst, [3, 2, 1, 0], i)
    #     sparse_accs.append(sparse_acc_arr)
    #     sparse_weights.append(sparse_weights_arr)


    # %%
    acc_arr, weights_arr, settings_arr = run_adaptive_model(trial_lst, label_lst, enabled_settings=[0, 1, 2, 3], train_setting=3, selection=0, sim_weights=None)
