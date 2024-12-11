from .utils import *
from .Model_Base import DCAGCell


class DCAG_model(nn.Module):
    def __init__(self, node_num, seq_len, graph_dim, blstm_dim, atten_head, DBN_pre,
                 choice=[1, 1, 1]):
        super(DCAG_model, self).__init__()
        self.pre = DBN_pre
        self.node_num = node_num
        self.seq_len = seq_len
        self.blstm_dim = blstm_dim
        self.graph_dim = graph_dim
        # self.output_dim = seq_len + np.sum(choice) * graph_dim
        self.output_dim = np.sum(choice) * graph_dim
        self.DCAGCell = DCAGCell(node_num, seq_len, graph_dim, blstm_dim, choice,atten_head,DBN_pre)
        self.output_linear1 = nn.Linear(in_features=2*(graph_dim+blstm_dim), out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, X):

        Feature_X = extract_features(X)

        bs, tlen, dimension = Feature_X.size()

        T_X = torch.reshape(X, [bs, tlen, dimension ])
        F_X = torch.reshape(X, [bs, dimension, tlen])
        T_edge_index = get_TA(T_X).to(T_X.device)
        F_edge_index = get_FA(F_X).to(F_X.device)

        DC_output, pre_output = self.DCAGCell(T_X, F_X, T_edge_index, F_edge_index)
        if self.pre==False:
            DC_output = self.relu(DC_output)
            DC_output = self.dropout(DC_output)
            DC_output = self.output_linear1(DC_output)

        return DC_output,pre_output
