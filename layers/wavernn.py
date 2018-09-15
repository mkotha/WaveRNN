import torch
import torch.nn as nn
import torch.nn.functional as F

def filter_none(xs):
    return [x for x in xs if x is not None]

class WaveRNN(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, feat_dims, aux_dims):
        super().__init__()
        self.n_classes = 256
        self.rnn_dims = rnn_dims
        self.aux_dims = aux_dims
        self.half_rnn_dims = rnn_dims // 2
        self.gru = nn.GRU(feat_dims + self.aux_dims + 3, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(self.half_rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, self.n_classes)
        self.fc3 = nn.Linear(self.half_rnn_dims + self.aux_dims, fc_dims)
        self.fc4 = nn.Linear(fc_dims, self.n_classes)

        coarse_mask = torch.cat([torch.ones(self.half_rnn_dims, feat_dims + self.aux_dims + 2), torch.zeros(self.half_rnn_dims, 1)], dim=1)
        i2h_mask = torch.cat([coarse_mask, torch.ones(self.half_rnn_dims, feat_dims + self.aux_dims + 3)], dim=0)
        self.mask = torch.cat([i2h_mask, i2h_mask, i2h_mask], dim=0).cuda().half()

    def forward(self, x, feat, aux1, aux2, aux3) :
        x = torch.cat(filter_none([feat, aux1, x]), dim=2)
        h, _ = self.gru(x)

        h_c, h_f = torch.split(h, self.half_rnn_dims, dim=2)

        o_c = F.relu(self.fc1(torch.cat(filter_none([h_c, aux2]), dim=2)))
        p_c = F.log_softmax(self.fc2(o_c), dim=2)
        #print(f'o_c: {o_c.var()} p_c: {p_c.var()}')

        o_f = F.relu(self.fc3(torch.cat(filter_none([h_f, aux3]), dim=2)))
        p_f = F.log_softmax(self.fc4(o_f), dim=2)
        #print(f'o_f: {o_f.var()} p_f: {p_f.var()}')

        return (p_c, p_f)

    def after_update(self):
        with torch.no_grad():
            self.gru.weight_ih_l0.data.mul_(self.mask)

    def to_cell(self):
        return WaveRNNCell(self.gru, self.rnn_dims,
                self.fc1, self.fc2, self.fc3, self.fc4)

class WaveRNNCell(nn.Module):
    def __init__(self, gru, rnn_dims, fc1, fc2, fc3, fc4):
        super().__init__()
        self.gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        self.gru_cell.weight_hh.data = gru.weight_hh_l0.data
        self.gru_cell.weight_ih.data = gru.weight_ih_l0.data
        self.gru_cell.bias_hh.data = gru.bias_hh_l0.data
        self.gru_cell.bias_ih.data = gru.bias_ih_l0.data
        self.rnn_dims = rnn_dims
        self.half_rnn_dims = rnn_dims // 2
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3
        self.fc4 = fc4

    def forward_c(self, x, feat, aux1, aux2, h):
        # print(f'x: {x.size()}, feat: {feat.size()}, aux1: {aux1.size()}')
        h_0 = self.gru_cell(torch.cat(filter_none([feat, aux1, x]), dim=1), h)
        h_c, _ = torch.split(h_0, self.half_rnn_dims, dim=1)
        return self.fc2(F.relu(self.fc1(torch.cat(filter_none([h_c, aux2]), dim=1))))

    def forward_f(self, x, feat, aux1, aux3, h):
        h_1 = self.gru_cell(torch.cat(filter_none([feat, aux1, x]), dim=1), h)
        _, h_f = torch.split(h_1, self.half_rnn_dims, dim=1)
        o_f = self.fc4(F.relu(self.fc3(torch.cat(filter_none([h_f, aux3]), dim=1))))
        return (o_f, h_1)
