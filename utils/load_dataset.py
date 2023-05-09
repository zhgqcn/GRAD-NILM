import os 
import numpy as np
from utils.ukdale_config import *


base_path = "/home/hrw/NILM/GRAD-NILM-TKDE/data/UKDALE"
train_path = os.path.join(base_path, "Train")
valid_path = os.path.join(base_path, "Valid")
test_path  = os.path.join(base_path, "Test")


def load_data(dataset, width, strides, set_type):

    ukdale_appliance_data = getUkdaleAppliancesAttribution()
    aggregate_mean, aggregate_std, aggregate_cutoff = getUkdaleAggregateAttribution()

    def import_data(app_type, set_type):
        path = dataset[set_type]
        # x = np.load(os.path.join(path, "Mains_house_GT_1_input.npy"))     # 真实
        x = np.load(os.path.join(path, "Mains_house_1_input.npy"))          # 合成
        y = np.load(os.path.join(path, "{}_appliance_house_1_target.npy".format(app_type.title())))
        s = np.load(os.path.join(path, "{}_appliance_house_1_state.npy".format(app_type.title())))
        print('\n{}'.format(app_type))
        print("Load {:5s} Data  ===> input:{:5d} target:{:5d} state:{:5d}".format(set_type.title(), len(x), len(y), len(s)))
        return x, y, s

    def seq_dataset(x, y, s, width, stride):
        x_ = []
        y_ = []
        s_ = []
        for t in range(0, x.shape[0]-width, stride):
            x_.append(x[t : t + width])
            y_.append(y[t : t + width])
            s_.append(s[t : t + width])

        x_ = np.array(x_).reshape([-1, width, 1])
        y_ = np.array(y_).reshape([-1, width, 1])
        s_ = np.array(s_).reshape([-1, width, 1])
        return x_, y_, s_

    def create_dataset(appliance, width, strides, set_type):
        x_tot = np.array([]).reshape(0, width, 1)
        y_tot = np.array([]).reshape(0, width, 1)
        s_tot = np.array([]).reshape(0, width, 1)

        x, y, s = import_data(appliance, set_type)        # Load complete dataset
        x_, y_, s_ = seq_dataset(x, y, s, width, strides) # Divide dataset in window

        print("Total Dataset    ===> x:{}, y:{}".format(x_.shape, y_.shape))

        x_tot = np.vstack([x_tot, x_])
        y_tot = np.vstack([y_tot, y_])
        s_tot = np.vstack([s_tot, s_])

        print("Complete Dataset ===> x:{}, y:{}, s:{}".format(x_tot.shape, y_tot.shape, s_tot.shape))

        return x_tot, y_tot, s_tot

    ###############################################################################
    # Load dataset
    ###############################################################################
    if dataset["name"] == "ukdale":
        print("###############################################################################")
        print("Create {:10s} dataset".format(set_type))
        Mains, W, Ws = create_dataset('WashingMachine', width, strides, set_type)
        Mains, D, Ds = create_dataset('Dishwasher',     width, strides, set_type)
        Mains, K, Ks = create_dataset('Kettle',         width, strides, set_type)
        Mains, F, Fs = create_dataset('Fridge',         width, strides, set_type)
        Mains, M, Ms = create_dataset('Microwave',      width, strides, set_type)

        W = normalize(W, ukdale_appliance_data['washingmachine']['mean'], ukdale_appliance_data['washingmachine']['std'])
        D = normalize(D, ukdale_appliance_data['dishwasher']['mean'], ukdale_appliance_data['dishwasher']['std'])
        K = normalize(K, ukdale_appliance_data['kettle']['mean'], ukdale_appliance_data['kettle']['std'])
        F = normalize(F, ukdale_appliance_data['fridge']['mean'], ukdale_appliance_data['fridge']['std'])
        M = normalize(M, ukdale_appliance_data['microwave']['mean'], ukdale_appliance_data['microwave']['std'])
        Mains = normalize(Mains, aggregate_mean, aggregate_std)

        targets = np.concatenate([W, D, K, F, M], axis=-1)
        states  = np.concatenate([Ws, Ds, Ks, Fs, Ms], axis=-1)

        del W, D, K, F, M, Ws, Ds, Ks, Fs, Ms

        return Mains, targets, states


def normalize(signal, mean, std):
    """
    Takes Signal and normalizes is by its mean and std

    Args:
        signal (np.array): signal to create quantiles
    Returns:
        normalized_signal (np.array): normalized signal
    """
    normalized_signal = (signal - mean) / std

    return normalized_signal


def reconstruct(y, width, strides, merge_type="mean"):
    
    len_total = width + (y.shape[0] - 1) * strides
    depth = width // strides
    
    yr = np.zeros([len_total, depth])
    yr[:] = np.nan
    
    for i in range(y.shape[0]):
        for d in range(depth):
            yr[i*strides+(d*strides):i*strides+((d+1)*strides),d] = y[i, d * strides : (d+1) * strides, 0]
    
    if merge_type == "mean":
        yr = np.nanmean(yr, axis=1)
    else:
        yr = np.nanmedian(yr, axis=1)
    
    return yr