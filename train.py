import torch
import random
import numpy as np
import pandas as pd
from utils.base import *
from utils.load_dataset import *
from utils.ukdale_config import *
from models.grad_nilm import *
import json, logging, os, datetime, argparse
import torch.utils.data as tud
import matplotlib.pyplot as plt
from torch.utils.data.dataset import TensorDataset
from sklearn.model_selection import train_test_split

###############################################################################
# Config
###############################################################################
seed_torch(seed=42)                                       # Fix the random seed
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--config", default="/home/hrw/NILM/GRAD-NILM-TKDE/config/grad_nilm.json", type=str, help="Path to the config file")
parser.add_argument("--detail", type=str, help="Description of the experiment")
a = parser.parse_args()

print("###############################################################################")
print("NILM DISAGREGATOR")
print("GPU : {}".format(a.gpu))
print("CONFIG : {}".format(a.config))
print("###############################################################################")

with open(a.config) as data_file:
    nilm = json.load(data_file)

for r in range(1, nilm["run"] + 1):
    ###############################################################################
    # Load dataset
    ###############################################################################
    x_train, y_train, s_train = load_data(nilm["dataset"], nilm["preprocessing"]["width"], nilm["preprocessing"]["strides"], set_type="train")
    x_valid, y_valid, s_valid = load_data(nilm["dataset"], nilm["preprocessing"]["width"], nilm["preprocessing"]["strides"], set_type="valid")

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(s_train).float())
    train_loader = tud.DataLoader(train_dataset, batch_size = nilm["training"]["batch_size"], shuffle = True, drop_last = True)

    valid_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).float(), torch.from_numpy(s_valid).float())
    valid_loader = tud.DataLoader(valid_dataset, batch_size = nilm["training"]["batch_size"], shuffle = False, drop_last = True)

    # Summary of all parameters
    print("###############################################################################")
    print("{}".format(nilm))
    print("Run number : {}/{}".format(r, nilm["run"]))
    print("###############################################################################")
    
    name = "experiments"
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists("{}/{}/{}/{}".format(name, nilm["dataset"]["name"], nilm["model"], time)):
        os.makedirs("{}/{}/{}/{}".format(name, nilm["dataset"]["name"], nilm["model"], time))

    with open("{}/{}/{}/{}/config.txt".format(name, nilm["dataset"]["name"], nilm["model"], time), "w") as outfile:
        json.dump(nilm, outfile)

    ###############################################################################
    # Training parameters
    ###############################################################################
    a.enable_lr_schedule = True
    a.num_epochs = nilm["training"]["epoch"]
    a.optimizer  = nilm["training"]["optimizer"]
    a.lr         = nilm["training"]["lr"]
    a.patience   = nilm["training"]["patience"]
    a.decay_step = nilm['training']['decay_steps']
    a.gamma      = nilm['training']['gamma']
    a.momentum   = None
    a.export_root = "{}/{}/{}/{}/{}".format(name, nilm["dataset"]["name"], nilm["model"], time, r)
    # a.e = "{}-{}-{}_House_{}-{}/{}".format(name, nilm["dataset"]["name"], nilm["model"], nilm["dataset"]["test"]["house"][0], time, r)
    a.train_means = torch.from_numpy(getMeanStd()[0])[:-1]
    a.train_stds  = torch.from_numpy(getMeanStd()[1])[:-1]
    a.weight_decay = 0.
    ###############################################################################
    # Create model and Train
    ###############################################################################
        
    if nilm["model"] == "GRAD-NILM":
        print('\nGRAD-NILM')
        model = GRADNILM()
        Trainer = GRADNILM_Trainer(a, model, train_loader, valid_loader)
        Trainer.train()

    else:
        exit()