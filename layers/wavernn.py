import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import utils.logger as logger

def filter_none(xs):
    return [x for x in xs if x is not None]

class WaveRNN(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, feat_dims, aux_dims):
        super().__init__()
        self.n_classes = 256
        self.rnn_dims = rnn_dims
        self.aux_dims = aux_dims
        self.half_rnn_dims = rnn_dims // 2
        self.feat_dims = feat_dims
        self.gru = nn.GRU(feat_dims + self.aux_dims + 3, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(self.half_rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, self.n_classes)
        self.fc3 = nn.Linear(self.half_rnn_dims + self.aux_dims, fc_dims)
        self.fc4 = nn.Linear(fc_dims, self.n_classes)
        self.register_buffer('mask', self.create_mask())

    def create_mask(self):
        coarse_mask = torch.cat([torch.ones(self.half_rnn_dims, self.feat_dims + self.aux_dims + 2), torch.zeros(self.half_rnn_dims, 1)], dim=1)
        i2h_mask = torch.cat([coarse_mask, torch.ones(self.half_rnn_dims, self.feat_dims + self.aux_dims + 3)], dim=0)
        return torch.cat([i2h_mask, i2h_mask, i2h_mask], dim=0)

    def forward(self, x, feat, aux1=None, aux2=None, aux3=None) :
        x = torch.cat(filter_none([feat, aux1, x]), dim=2)
        h, h_n = self.gru(x)

        h_c, h_f = torch.split(h, self.half_rnn_dims, dim=2)

        o_c = F.relu(self.fc1(torch.cat(filter_none([h_c, aux2]), dim=2)))
        p_c = F.log_softmax(self.fc2(o_c), dim=2)
        #logger.log(f'o_c: {o_c.var()} p_c: {p_c.var()}')

        o_f = F.relu(self.fc3(torch.cat(filter_none([h_f, aux3]), dim=2)))
        p_f = F.log_softmax(self.fc4(o_f), dim=2)
        #logger.log(f'o_f: {o_f.var()} p_f: {p_f.var()}')

        return p_c, p_f, h_n.squeeze(0)

    def after_update(self):
        with torch.no_grad():
            self.gru.weight_ih_l0.data.mul_(self.mask)

    def to_cell(self):
        return WaveRNNCell(self.gru, self.rnn_dims,
                self.fc1, self.fc2, self.fc3, self.fc4)

    def generate(self, feat, aux1=None, aux2=None, aux3=None, deterministic=False, use_half=False, verbose=False, seq_len=None, batch_size=None):
        start = time.time()
        if seq_len is None:
            seq_len = feat.size(1)
        if batch_size is None:
            batch_size = feat.size(0)
        h = torch.zeros(batch_size, self.rnn_dims).cuda()
        if use_half:
            h = h.half()

        c_val = torch.zeros(batch_size).cuda()
        f_val = torch.zeros(batch_size).cuda()
        zero = torch.zeros(batch_size).cuda()
        rnn_cell = self.to_cell()
        output = []

        for i in range(seq_len) :
            if feat is None:
                m_t = None
            else:
                m_t = feat[:, i, :]
            if aux1 is None:
                a1_t = None
            else:
                a1_t = aux1[:, i, :]
            if aux2 is None:
                a2_t = None
            else:
                a2_t = aux2[:, i, :]
            if aux3 is None:
                a3_t = None
            else:
                a3_t = aux3[:, i, :]

            x = torch.stack([c_val, f_val, zero], dim=1)
            if use_half:
                x = x.half()
            o_c = rnn_cell.forward_c(x, m_t, a1_t, a2_t, h)
            if deterministic:
                c_cat = torch.argmax(o_c, dim=1).to(torch.float32)
            else:
                posterior_c = F.softmax(o_c.float(), dim=1)
                distrib_c = torch.distributions.Categorical(posterior_c)
                c_cat = distrib_c.sample().float()
            c_val_new = c_cat / 127.5 - 1.0

            x = torch.stack([c_val, f_val, c_val_new], dim=1)
            if use_half:
                x = x.half()
            o_f, h = rnn_cell.forward_f(x, m_t, a1_t, a3_t, h)
            if deterministic:
                f_cat = torch.argmax(o_f, dim=1).to(torch.float32)
            else:
                posterior_f = F.softmax(o_f.float(), dim=1)
                distrib_f = torch.distributions.Categorical(posterior_f)
                f_cat = distrib_f.sample().float()
            f_val = f_cat / 127.5 - 1.0

            c_val = c_val_new

            sample = (c_cat * 256 + f_cat) / 32767.5 - 1.0
            if verbose and i % 10000 < 100:
                logger.log(f'c={c_cat[0]} f={f_cat[0]} sample={sample[0]}')
            output.append(sample)
            if i % 100 == 0 :
                speed = int((i + 1) / (time.time() - start))
                logger.status(f'{i+1}/{seq_len} -- Speed: {speed} samples/sec')

        return torch.stack(output, dim=1)

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
        # logger.log(f'x: {x.size()}, feat: {feat.size()}, aux1: {aux1.size()}')
        h_0 = self.gru_cell(torch.cat(filter_none([feat, aux1, x]), dim=1), h)
        h_c, _ = torch.split(h_0, self.half_rnn_dims, dim=1)
        return self.fc2(F.relu(self.fc1(torch.cat(filter_none([h_c, aux2]), dim=1))))

    def forward_f(self, x, feat, aux1, aux3, h):
        h_1 = self.gru_cell(torch.cat(filter_none([feat, aux1, x]), dim=1), h)
        _, h_f = torch.split(h_1, self.half_rnn_dims, dim=1)
        o_f = self.fc4(F.relu(self.fc3(torch.cat(filter_none([h_f, aux3]), dim=1))))
        return (o_f, h_1)
