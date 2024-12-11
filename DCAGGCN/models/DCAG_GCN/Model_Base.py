import numpy as np
import torch.utils.data
from .DBN import DBN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class BiLSTM(nn.Module):
    def __init__(self, input_size,hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,batch_first=True,bidirectional=True)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, input_seq.shape[0], self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, input_seq.shape[0], self.hidden_size).to(device)
        seq_len = input_seq.shape[1]
        output, _ = self.lstm(input_seq, (h_0, c_0))
        forward_hidden = output[:, -1, :self.hidden_size]
        backward_hidden = output[:, 0, self.hidden_size:]
        combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        return combined_hidden
class DCAGCell(nn.Module):
    def __init__(self, node_num, seq_len, graph_dim, blstm_dim, choice, atten_head, DBN_pre):
        super(DCAGCell, self).__init__()
        self.pre = DBN_pre
        self.node_num = node_num
        self.seq_len = seq_len
        self.graph_dim = graph_dim
        self.blstm_dim = blstm_dim
        self.output_dim = np.sum(choice) * graph_dim
        self.choice = choice
        self.Tseq_linear = nn.Linear(in_features=self.seq_len, out_features=self.graph_dim)
        self.Fseq_linear = nn.Linear(in_features=self.node_num, out_features=self.graph_dim)
        if choice[0] == 1:
            print(f"[Bi-LSTM]")
            self.self_atten = nn.MultiheadAttention(embed_dim=node_num, num_heads=atten_head)
            self.blstm = BiLSTM(input_size=self.seq_len, hidden_size=self.blstm_dim, num_layers=1)

        if choice[1] == 1:
            print(f"[Time-correlation]")
            self.TC_origin = nn.Linear(in_features=seq_len, out_features=graph_dim)
            self.TC_gconv1 = GATConv(seq_len, graph_dim, heads=3, concat=False)
            self.TC_gconv2 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.TC_gconv3 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.TC_gconv4 = GATConv(graph_dim, graph_dim, heads=1, concat=False)
            # self.TC_gconv5 = GATConv(graph_dim, graph_dim, heads = 1, concat = False)
            self.T_source_embed = nn.Parameter(torch.Tensor(self.node_num, 100))
            self.T_target_embed = nn.Parameter(torch.Tensor(100, self.node_num))
            self.TC_linear_1 = nn.Linear(self.seq_len, self.graph_dim)
            self.TC_linear_2 = nn.Linear(self.graph_dim, self.graph_dim)
            self.TC_linear_3 = nn.Linear(self.graph_dim, self.graph_dim)
            self.TC_linear_4 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.TC_linear_5 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.TC_jklayer = JumpingKnowledge("max")

            nn.init.xavier_uniform_(self.T_source_embed)
            nn.init.xavier_uniform_(self.T_target_embed)

        if choice[2] == 1:
            print(f"[Spatial-correlation]")
            self.SC_origin = nn.Linear(in_features=node_num, out_features=graph_dim)
            self.SC_gconv1 = GATConv(node_num, graph_dim, heads=3, concat=False)
            self.SC_gconv2 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.SC_gconv3 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            self.SC_gconv4 = GATConv(graph_dim, graph_dim, heads=3, concat=False)
            # self.SC_gconv5 = GATConv(graph_dim, graph_dim, heads = 1, concat = False)
            self.F_source_embed = nn.Parameter(torch.Tensor(self.seq_len, 100))
            self.F_target_embed = nn.Parameter(torch.Tensor(100, self.seq_len))
            self.SC_linear_1 = nn.Linear(self.node_num, self.graph_dim)
            self.SC_linear_2 = nn.Linear(self.graph_dim, self.graph_dim)
            self.SC_linear_3 = nn.Linear(self.graph_dim, self.graph_dim)
            self.SC_linear_4 = nn.Linear(self.graph_dim, self.graph_dim)
            self.dbn = DBN([(node_num + seq_len) * graph_dim, 400, 200, 40])
            # self.SC_linear_5 = nn.Linear(self.graph_dim, self.graph_dim)
            # self.SC_jklayer = JumpingKnowledge("max")

            nn.init.xavier_uniform_(self.F_source_embed)
            nn.init.xavier_uniform_(self.F_target_embed)

    def forward(self, T_x, F_x, edge_index, S_edge_index, pre):
        output_list = [[1], 0, 0]

        if self.choice[0] == 1:
            atten_input = torch.reshape(T_x, (-1, self.node_num, self.seq_len)).permute(2, 0, 1)
            atten_output, _ = self.self_atten(atten_input, atten_input, atten_input)
            atten_output = torch.tanh(atten_output + atten_input)
            atten_output = torch.reshape(atten_output, (atten_input.shape[1], self.node_num,-1))
            blstm_output = self.blstm(atten_output)
            output_list[0] = blstm_output

        if self.choice[1] == 1:
            batch, num, fea = T_x.size()
            T_x = torch.reshape(T_x, [batch * num, fea])
            # T_x = self.Tseq_linear(T_x) + T_x

            TC_learned_matrix = F.softmax(F.relu(torch.mm(self.T_source_embed, self.T_target_embed)), dim=1)

            TC_gout_1 = self.TC_gconv1(T_x, edge_index)
            adp_input_1 = torch.reshape(T_x, (-1, self.node_num, self.seq_len))
            TC_adp_1 = self.TC_linear_1(TC_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))
            TC_adp_1 = torch.reshape(TC_adp_1, (-1, self.graph_dim))
            TC_origin = self.TC_origin(T_x)
            TC_output_1 = torch.tanh(TC_gout_1) * torch.sigmoid(TC_adp_1) + TC_origin * (1 - torch.sigmoid(TC_adp_1))

            TC_gout_2 = self.TC_gconv2(torch.tanh(TC_output_1), edge_index)
            adp_input_2 = torch.reshape(torch.tanh(TC_output_1), (-1, self.node_num, self.graph_dim))
            TC_adp_2 = self.TC_linear_2(TC_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
            TC_adp_2 = torch.reshape(TC_adp_2, (-1, self.graph_dim))
            TC_output_2 = F.leaky_relu(TC_gout_2) * torch.sigmoid(TC_adp_2) + TC_output_1 * (
                    1 - torch.sigmoid(TC_adp_2))

            TC_gout_3 = self.TC_gconv3(F.relu(TC_output_2), edge_index)
            adp_input_3 = torch.reshape(F.relu(TC_output_2), (-1, self.node_num, self.graph_dim))
            TC_adp_3 = self.TC_linear_3(TC_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
            TC_adp_3 = torch.reshape(TC_adp_3, (-1, self.graph_dim))
            TC_output_3 = F.relu(TC_gout_3) * torch.sigmoid(TC_adp_3) + TC_output_2 * (1 - torch.sigmoid(TC_adp_3))


            TC_output = torch.reshape(F.relu(TC_output_3), (-1, self.node_num, self.graph_dim))

            output_list[1] = TC_output

        if self.choice[2] == 1:
            batch, num, fea = F_x.size()
            F_x = torch.reshape(F_x, [batch * num, fea])
            # F_x = self.Fseq_linear(F_x) + F_x

            SC_learned_matrix = F.softmax(F.relu(torch.mm(self.F_source_embed, self.F_target_embed)), dim=1)

            SC_gout_1 = self.SC_gconv1(F_x, S_edge_index)
            adp_input_1 = torch.reshape(F_x, (-1, self.seq_len, self.node_num))
            SC_adp_1 = self.SC_linear_1(SC_learned_matrix.matmul(F.dropout(adp_input_1, p=0.1)))
            SC_adp_1 = torch.reshape(SC_adp_1, (-1, self.graph_dim))
            SC_origin = self.SC_origin(F_x)
            SC_output_1 = torch.tanh(SC_gout_1) * torch.sigmoid(SC_adp_1) + SC_origin * (
                    1 - torch.sigmoid(SC_adp_1))

            SC_gout_2 = self.SC_gconv2(torch.tanh(SC_output_1), S_edge_index)
            adp_input_2 = torch.reshape(torch.tanh(SC_output_1), (-1, self.seq_len, self.graph_dim))
            SC_adp_2 = self.SC_linear_2(SC_learned_matrix.matmul(F.dropout(adp_input_2, p=0.1)))
            SC_adp_2 = torch.reshape(SC_adp_2, (-1, self.graph_dim))
            SC_output_2 = F.leaky_relu(SC_gout_2) * torch.sigmoid(SC_adp_2) + SC_output_1 * (
                    1 - torch.sigmoid(SC_adp_2))

            SC_gout_3 = self.SC_gconv3(F.relu(SC_output_2), S_edge_index)
            adp_input_3 = torch.reshape(F.relu(SC_output_2), (-1, self.seq_len, self.graph_dim))
            SC_adp_3 = self.SC_linear_3(SC_learned_matrix.matmul(F.dropout(adp_input_3, p=0.1)))
            SC_adp_3 = torch.reshape(SC_adp_3, (-1, self.graph_dim))
            SC_output_3 = F.relu(SC_gout_3) * torch.sigmoid(SC_adp_3) + SC_output_2 * (1 - torch.sigmoid(SC_adp_3))


            SC_output = torch.reshape(F.relu(SC_output_3), (-1, self.seq_len, self.graph_dim))

            output_list[2] = SC_output

        if self.choice[1]:
            dbn_input = output_list[1]
            if self.choice[2]:
                dbn_input = torch.cat((dbn_input, output_list[2]), dim=1)
        else:
            dbn_input = output_list[2]

        dbn_input = torch.reshape(dbn_input, (dbn_input.shape[0], -1))

        if self.DBN_or_not:
            pretrain_output = dbn_input
            if pre == False:
                for rbm in self.dbn.RBMs:
                    rbm.W = rbm.W.to(device)
                    rbm.h_bias = rbm.h_bias.to(device)
                    rbm.v_bias = rbm.v_bias.to(device)
                cell_output = torch.as_tensor(dbn_input, dtype=torch.double)
                dbn_output = self.dbn(cell_output)
                dbn_output = torch.as_tensor(dbn_output, dtype=torch.float32)
                if self.choice[0]:
                    out = torch.cat((dbn_output, blstm_output), dim=1)
                else:
                    out = dbn_output
            else:
                out = None
            return out, pretrain_output
        else:
            if self.choice[0]:
                return torch.cat((dbn_input, blstm_output), dim=1), None
            else:
                return dbn_input, None

