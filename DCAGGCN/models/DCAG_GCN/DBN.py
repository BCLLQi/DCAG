import torch
import torch.nn as nn
import numpy as np

def sigmoid(x):
    return 1. / (1 + torch.exp(-x))
class RBM(nn.Module):
    def __init__(self,n_visible=2, n_hidden=3, W=None, h_bias=None, v_bias=None, np_rng=None):
        super(RBM, self).__init__()

        if np_rng is None:
            np_rng = np.random.RandomState(2345)
            torch.manual_seed(1234)

        if W is None:
            a = 1. / n_visible
            initial_W = np.array(np_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if h_bias is None:
            h_bias = np.zeros(n_hidden)  # initialize h bias 0

        if v_bias is None:
            v_bias = np.zeros(n_visible)  # initialize v bias 0

        '''self.tc_rng = tc_rng'''
        self.W =torch.from_numpy(W).double()
        self.h_bias =torch.from_numpy(h_bias).double()
        self.v_bias =torch.from_numpy(v_bias).double()

    def sample_h(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = torch.bernoulli(h1_mean)
        h1_sample=torch.as_tensor(h1_sample, dtype=torch.double)
        return [h1_mean, h1_sample]

    def sample_v(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = torch.bernoulli(v1_mean)
        v1_sample=torch.as_tensor(v1_sample, dtype=torch.double)
        return [v1_mean, v1_sample]

    def free_energy(self, v):
        v_bias_term = torch.matmul(v, self.bv)
        wx_b = torch.matmul(v, self.W.t()) + self.bh
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        return -hidden_term - v_bias_term

    def propup(self, v):
        pre_sigmoid_activation = torch.matmul(v, self.W) + self.h_bias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = torch.matmul(h, self.W.T) + self.v_bias
        return sigmoid(pre_sigmoid_activation)


class DBN(nn.Module):
    def __init__(self, sizes):
        super(DBN, self).__init__()
        self.RBMs = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.RBMs.append(RBM(n_visible=sizes[i], n_hidden=sizes[i+1]))

    def pretrain(self,data,epochs):
        input = torch.as_tensor(data, dtype=torch.double)
        for index, rbm in enumerate(self.RBMs):
            if index != 0:
                _, input = self.RBMs[index - 1].sample_h(input)
            for epoch in range(epochs):
                if index == 0:
                    contrastive_divergence(rbm=rbm, data=data, learning_rate=0.1)
                else:
                    contrastive_divergence(rbm=rbm, data=input, learning_rate=0.1)
            #print(f"RBM {index} trained.")

    def forward(self, x):
        for rbm in self.RBMs:
            _,h = rbm.sample_h(x)
            x = h
        return x


def contrastive_divergence(rbm, data, learning_rate):
    v0 = torch.as_tensor(data, dtype=torch.double)
    h0_prob, h0_sample = rbm.sample_h(v0)
    v1_prob, _ = rbm.sample_v(h0_sample)
    h1_prob, _ = rbm.sample_h(v1_prob)

    positive_grad = torch.matmul(h0_prob.T, v0)
    negative_grad = torch.matmul(h1_prob.T, v1_prob)
    rbm.W += (learning_rate * (positive_grad - negative_grad) / data.size(0)).T
    rbm.v_bias += learning_rate * torch.mean(v0 - v1_prob, dim=0)
    rbm.h_bias += learning_rate * torch.mean(h0_prob - h1_prob, dim=0)
