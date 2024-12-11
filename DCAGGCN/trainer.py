import torch
import torch.nn.functional as F

import os
# import wandb
import pandas as pd
import numpy as np
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class

from utils import fix_randomness, starting_logs, save_checkpoint, _calc_metrics_bearing, _calc_metrics
import warnings

import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import collections
from algorithms.algorithms import get_algorithm_class
from utils import AverageMeter

torch.backends.cudnn.benchmark = True



class GNN_RUL_trainer(object):

    def __init__(self, args):
        self.GNN_method = args.GNN_method
        self.dataset = args.dataset
        self.dataset_id = args.dataset_id
        self.device = torch.device(args.device)
        self.bearing_id = args.bearing_id

        self.run_description = args.run_description
        self.experiment_description = args.experiment_description


        self.data_path = os.path.join(args.data_path, self.dataset, self.dataset_id)

        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.create_save_dir()

        self.num_runs = args.num_runs

        self.dataset_configs, self.hparams_class = self.get_configs(args.dataset_id)

        self.train_configs = self.hparams_class.train_params[self.GNN_method]
        self.model_configs = self.hparams_class.alg_hparams[self.GNN_method]

        self.default_hparams = {**self.hparams_class.alg_hparams[self.GNN_method],
                                **self.hparams_class.train_params[self.GNN_method]}


    def train(self):
        run_name = f"{self.run_description}"

        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)


        for run_id in range(self.num_runs):
            fix_randomness(run_id)

            self.logger, self.log_dir = starting_logs(self.dataset, self.GNN_method, self.exp_log_dir,self.dataset_id, self.bearing_id, run_id)

            self.train_dl, self.test_dl, self.max_ruls = data_generator(self.data_path,  self.dataset_configs, self.train_configs)
            if isinstance(self.test_dl, dict):
                self.best_result = dict()
                for key in self.test_dl.keys():
                    self.best_result[key] = [[np.Inf], [np.Inf], [np.Inf], [np.Inf]]
            else:
                self.best_result = [[np.Inf], [np.Inf], [np.Inf], [np.Inf]]

            algorithm_class = get_algorithm_class(self.GNN_method)
            algorithm = algorithm_class(self.model_configs, self.train_configs, self.device)
            algorithm.to(self.device)

            loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

            for epoch in range(1, self.train_configs["num_epochs"] + 1):
                algorithm.train()

                for step, (X, y) in enumerate(self.train_dl):
                    algorithm.pre_train(X)

                for step, (X, y) in enumerate(self.train_dl):
                    X, y = X.float().to(self.device), y.float().to(self.device)

                    losses = algorithm.update(X, y, epoch)
                    for key, val in losses.items():
                        loss_avg_meters[key].update(val, X.size(0))

                self.logger.debug(f'[Epoch : {epoch}/{self.train_configs["num_epochs"]}]')
                for key, val in loss_avg_meters.items():
                    self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                self.test_prediction(algorithm)

                self.logger.debug(f'-------------------------------------')

            self.algorithm = algorithm
            save_checkpoint(self.home_path, self.algorithm, self.dataset_configs,
                            self.log_dir, self.default_hparams)


    def test_base(self, model, test_dataloader):
        pred_labels = np.array([])
        true_labels = np.array([])
        loss_total = []
        with torch.no_grad():
            for data, labels in test_dataloader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).float().to(self.device)

                predictions = model(data)
                predictions = predictions.view((-1))

                loss = F.mse_loss(predictions, labels)
                loss_total.append(loss.item())
                pred = predictions.detach()

                pred_labels = np.append(pred_labels, pred.cpu().numpy())
                true_labels = np.append(true_labels, labels.cpu().numpy())
        return pred_labels, true_labels, loss_total

    def test_prediction(self,algorithm):
        model = algorithm.model.to(self.device)

        model.eval()

        if isinstance(self.test_dl, dict):

            test_pre = dict()
            test_real = dict()
            test_total_loss = dict()

            for key, test_dataloader_i in self.test_dl.items():
                pre_i, real_i, loss_i = self.test_base(model, test_dataloader_i)

                test_pre[key] = pre_i
                test_real[key] = real_i
                test_total_loss[key] = torch.tensor(loss_i).mean()

        else:
            test_pre, test_real, test_total_loss = self.test_base(model, self.test_dl)
            test_total_loss = torch.tensor(test_total_loss).mean()

        self.pred_labels = test_pre
        self.true_labels = test_real
        self.total_loss = test_total_loss


    def get_configs(self, dataset_id):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)

        dataset = dataset_class()
        hparams = hparams_class(dataset_id)

        return dataset, hparams


    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

