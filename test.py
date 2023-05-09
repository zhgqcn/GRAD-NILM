import torch
import random
import numpy as np
import pandas as pd
from ast import Raise
from utils.base import *
from utils.metrics import *
from models.grad_nilm import *
from utils.load_dataset import *
from utils.ukdale_config import *
import torch.utils.data as tud
import json, logging, os, datetime, argparse
from torch.utils.data.dataset import TensorDataset

###############################################################################
# Config
###############################################################################
seed_torch(seed=42)                                       # Fix the random seed
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int, help="GPU to use")
parser.add_argument("--config", default="/home/hrw/NILM/GRAD-NILM-TKDE/experiments/ukdale/GRAD-NILM/20220911-140636/config.txt", type=str, help="Path to the config file")
a = parser.parse_args()

with open(a.config) as data_file:
    nilm = json.load(data_file)

epochs = nilm["training"]["epoch"]
start = nilm["training"]["start_stopping"]

print("###############################################################################")
print("NILM DISAGREGATOR")
print("GPU : {}".format(a.gpu))
print("CONFIG : {}".format(a.config))
print("###############################################################################")

app_list = nilm["appliance"]
width = nilm["preprocessing"]["width"]
stride = nilm["preprocessing"]["strides"]

MAE_run    = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
MAE_run_on = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
EpD_run    = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
ACC_run    = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
PR_run     = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
RE_run     = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
F1_run     = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
SAE_run    = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
RMSE_run   = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
EAC_run    = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}
NDE_run    = {'WashingMachine': [], 'Dishwasher': [], 'Kettle': [], 'Fridge': [], 'Microwave': []}

# Load Data
a.device = 'cuda'
a.gamma  = 0.5
a.weight_decay = 0.
a.num_epochs   = nilm["training"]["epoch"]
a.optimizer    = nilm["training"]["optimizer"]
a.lr           = nilm["training"]["lr"]
a.patience     = nilm["training"]["patience"]
a.decay_step   = nilm["training"]["decay_steps"]
a.enable_lr_schedule = True

a.train_means = torch.from_numpy(getMeanStd()[0])[:-1]
a.train_stds  = torch.from_numpy(getMeanStd()[1])[:-1]
x_test, y_test, s_test = load_data(nilm["dataset"], nilm["preprocessing"]["width"], nilm["preprocessing"]["strides"], set_type="test")
test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(s_test).float())
test_loader = tud.DataLoader(test_dataset, batch_size = 512, shuffle = False, num_workers = 0, drop_last = True)

for r in range(1, nilm["run"] + 1):    

    if nilm["model"] == "GRAD-NILM":
        time = a.config.split('/')[-2]
        a.export_root = "/home/hrw/NILM/GRAD-NILM-TKDE/experiments/{}/{}/{}/{}".format(nilm["dataset"]["name"], nilm["model"], time, r)
        model = GRADNILM()
        Trainer = GRADNILM_Trainer(a, model, None, None)
        y_pred, s_pred, x_true, y_true, s_true = Trainer.test(test_loader)
    else:
        Raise()

    for index_app, test_app in enumerate(['WashingMachine', 'Dishwasher', 'Kettle', 'Fridge', 'Microwave']):
        x_total = []
        y_total_pred = []
        y_total_true = []
        x_all        = reconstruct(x_true * getUkdaleAggregateAttribution()[0] + getUkdaleAggregateAttribution()[1], width, stride, 'median')
        y_all_pred_w = reconstruct(np.expand_dims(y_pred[:,:,index_app], axis=-1), width, stride, 'median')
        y_all_true_w = reconstruct(np.expand_dims(y_true[:,:,index_app], axis=-1), width, stride, 'median')
        s_all_pred_w = reconstruct(np.expand_dims(s_pred[:,:,index_app], axis=-1), width, stride, 'median')
        s_all_true_w = reconstruct(np.expand_dims(s_true[:,:,index_app], axis=-1), width, stride, 'median')
        s_all_pred_w[s_all_pred_w  > 0.] = 1
        s_all_pred_w[s_all_pred_w  < 0.] = 0

        y_all_pred_w = y_all_pred_w * s_all_pred_w
        y_all_pred_w[y_all_pred_w < 5] = 0

        y_all_true_w = y_all_true_w * s_all_true_w
        y_all_true_w[y_all_true_w < 5] = 0
        
        x_all = x_all.reshape([1,-1])
        y_all_pred = y_all_pred_w.reshape([1,-1])
        y_all_true = y_all_true_w.reshape([1,-1])

        ###############################################################################
        # Completed sequence
        ###############################################################################
        for i in range(x_all.shape[-1]):
            x_total.append(x_all[0, i])
        for i in range(y_all_pred.shape[-1]):    
            y_total_pred.append(y_all_pred[0, i])
        for i in range(y_all_true.shape[-1]):
            y_total_true.append(y_all_true[0, i])

        #print(len(x_total))
        
        del x_all
        del y_all_pred
        del y_all_true
        
        ###############################################################################
        # Transform in array
        ###############################################################################
        x_total = np.array(x_total).reshape([1,-1])
        y_total_pred = np.array(y_total_pred).reshape([1,-1])
        y_total_true = np.array(y_total_true).reshape([1,-1])

        np.save("/home/hrw/NILM/GRAD-NILM-TKDE/experiments/ukdale/{}/{}/pred_{}_{}.npy".format(nilm["model"], time, r, test_app),\
                [x_total, y_total_pred, y_total_true])

        print(f'===== {test_app} ========')
        MAE_tot, MAE_app, MAE          = MAE_metric(y_total_pred, y_total_true, disaggregation=True, only_power_on=False)
        MAE_tot_on, MAE_app_on, MAE_on = MAE_metric(y_total_pred, y_total_true, disaggregation=True, only_power_on=True)
        EpD_app = EpD_metric(y_total_pred, y_total_true)
        acc_P_tot, acc_P_app, acc_P = acc_Power(y_total_pred, y_total_true, disaggregation=True)
        thr = getUkdaleAppliancesAttribution()[test_app.lower()]['on_power_threshold']
        PR_app = PR_metric(y_total_pred, y_total_true, thr)
        RE_app = RE_metric(y_total_pred, y_total_true, thr)
        F1_app = F1_metric(y_total_pred, y_total_true, thr)
        SAE_app = SAE_metric(y_total_pred, y_total_true)
        RMSE_app = RMSE_metric(y_total_pred, y_total_true)
        EAC_app = EAC_metric(y_total_pred, y_total_true)
        NDE_app = NDE_metric(y_total_pred, y_total_true)

        if (np.isnan(acc_P_tot)) or (F1_app[0] == 0):
            print("Error Detected")
        else:
            MAE_run[test_app].append(MAE_tot)
            MAE_run_on[test_app].append(MAE_tot_on)
            EpD_run[test_app].append(EpD_app)
            ACC_run[test_app].append(acc_P_tot)
            PR_run[test_app].append(PR_app[0])
            RE_run[test_app].append(RE_app[0])
            F1_run[test_app].append(F1_app[0])
            SAE_run[test_app].append(SAE_app[0])
            RMSE_run[test_app].append(RMSE_app)
            EAC_run[test_app].append(EAC_app)
            NDE_run[test_app].append(NDE_app)

for index_app, test_app in enumerate(['WashingMachine', 'Dishwasher', 'Kettle', 'Fridge', 'Microwave']):
    print(f'=========== {test_app} ===================')
    print('MAE     :', np.mean(MAE_run[test_app]), np.std(MAE_run[test_app]))
    print('MAE_on  :', np.mean(MAE_run_on[test_app]), np.std(MAE_run_on[test_app]))
    print('EpD     :', np.mean(EpD_run[test_app]), np.std(EpD_run[test_app]))
    print('PR      :', np.mean(PR_run[test_app]), np.std(PR_run[test_app]))
    print('RE      :', np.mean(RE_run[test_app]), np.std(RE_run[test_app]))
    print('F1      :', np.mean(F1_run[test_app]), np.std(F1_run[test_app]))
    print('Acc     :', np.mean(ACC_run[test_app]), np.std(ACC_run[test_app]))
    print('SAE     :', np.mean(SAE_run[test_app]), np.std(SAE_run[test_app]))
    print('RMSE    :', np.mean(RMSE_run[test_app]), np.std(RMSE_run[test_app]))
    print('EAC     :', np.mean(EAC_run[test_app]), np.std(EAC_run[test_app]))
    print('NDE     :', np.mean(NDE_run[test_app]), np.std(NDE_run[test_app]))

    np.save("/home/hrw/NILM/GRAD-NILM-TKDE/experiments/ukdale/{}/{}/results_median_{}.npy".format(nilm["model"], time, test_app),\
                                         [MAE_run[test_app], MAE_run_on[test_app], EpD_run[test_app], PR_run[test_app], \
                                          RE_run[test_app], F1_run[test_app], ACC_run[test_app], SAE_run[test_app], \
                                          RMSE_run[test_app], EAC_run[test_app], NDE_run[test_app]])
