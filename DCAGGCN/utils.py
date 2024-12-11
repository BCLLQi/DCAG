import torch
import torch.nn.functional as F
from torch import nn as nn

import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from thop import profile
from algorithms.algorithms import get_algorithm_class


def complexity_computation(model, dataset_configs, device):

    x = torch.rand(1, dataset_configs.input_channels, dataset_configs.sequence_len).to(device)
    x = x.squeeze(1)

    macs, params = profile(model, inputs=(x,))
    flops = 2*macs

    return flops, params



def model_complexity(GNN_method, model_configs, training_configs, dataset_configs, device):
    algorithm_class = get_algorithm_class(GNN_method)
    algorithm = algorithm_class(model_configs, training_configs, device)
    algorithm.to(device)

    model = algorithm.model

    flops, params = complexity_computation(model, dataset_configs, device)

    return flops, params



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def _logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger



def starting_logs(data_type, GNN_method, exp_log_dir, dataset_id, bearing_id, run_id):
    log_dir = os.path.join(exp_log_dir, GNN_method + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')

    if data_type =='PHM2012' or data_type == 'XJTU_SY':
        logger.debug(f'Sub-dataset ID:  {dataset_id}')

    logger.debug(f'Method:  {GNN_method}')

    logger.debug("=" * 45)
    logger.debug(f'Run ID: {run_id}')
    logger.debug("=" * 45)
    return logger, log_dir



def save_checkpoint(home_path, algorithm, dataset_configs, log_dir, hparams):

    save_dict = {
        "configs": dataset_configs.__dict__,
        "hparams": dict(hparams),
        "model_dict": algorithm.state_dict()
    }
    # save classification report
    save_path = os.path.join(home_path, log_dir, "checkpoint.pt")
    #torch.save(save_dict, save_path)


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)




def rmse_value(predicted, real):
    MSE = mean_squared_error(real,predicted)
    RMSE = np.sqrt(MSE)
    return RMSE


def mae_value(predicted, real):
    MAE = mean_absolute_error(real,predicted)
    return MAE



def _calc_metrics_bearing(pred_labels, true_labels):
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)

    RMSE = rmse_value(pred_labels, true_labels)
    MAE = mae_value(pred_labels, true_labels)

    return  MAE, RMSE


def _calc_metrics(pred_labels, true_labels):
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)

    RMSE = rmse_value(pred_labels, true_labels)
    MAE = mae_value(pred_labels, true_labels)


    return  MAE, RMSE