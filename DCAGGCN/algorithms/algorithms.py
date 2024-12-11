import torch
import torch.nn as nn
import numpy as np
from models.DCAG_GCN.Model import DCAG_model

def get_algorithm_class(algorithm_name):

    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))

    return globals()[algorithm_name]



class Algorithm(torch.nn.Module):

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.mse = nn.MSELoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class DCAG_GCN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(DCAG_GCN, self).__init__(configs)

        # print(configs)

        self.model = DCAG_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch=None):
        predicted_RUL,_ = self.model(X)

        loss = self.mse(predicted_RUL, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def pre_train(self,X):
        _,pretrain_output=self.model(X)
        self.model.DCAGCell.dbn.pretrain(pretrain_output,5)

