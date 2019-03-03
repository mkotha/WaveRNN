import torch
import torch.nn as nn
import math
from layers.wavernn import WaveRNN
import utils.logger as logger
import utils.nn
import time

class Conv2(nn.Module):
    """ A convolution layer with the stride of 2.

        Input:
            x: (N, 2L+2, in_channels) numeric tensor
            global_cond: (N, global_cond_channels) numeric tensor
        Output:
            y: (N, L, out_channels) numeric tensor
    """
    def __init__(self, in_channels, out_channels, global_cond_channels):
        super().__init__()

        ksz = 4
        self.out_channels = out_channels
        if 0 < global_cond_channels:
            self.w_cond = nn.Linear(global_cond_channels, 2 * out_channels, bias=False)
        self.conv_wide = nn.Conv1d(in_channels, 2 * out_channels, ksz, stride=2)
        wsize = 2.967 / math.sqrt(ksz * in_channels)
        self.conv_wide.weight.data.uniform_(-wsize, wsize)
        self.conv_wide.bias.data.zero_()

    def forward(self, x, global_cond):
        x1 = self.conv_wide(x.transpose(1, 2)).transpose(1, 2)
        if global_cond is not None:
            x2 = self.w_cond(global_cond).unsqueeze(1).expand(-1, x1.size(1), -1)
        else:
            x2 = torch.zeros_like(x1)
        a, b = (x1 + x2).split(self.out_channels, dim=2)
        return torch.sigmoid(a) * torch.tanh(b)

class Conv4(nn.Module):
    """ A convolution layer with the stride of 4.

        Input:
            x: (N, 4L+6, in_channels) numeric tensor
            global_cond: (N, global_cond_channels) numeric tensor
        Output:
            y: (N, L, out_channels) numeric tensor
    """
    def __init__(self, in_channels, out_channels, global_cond_channels):
        super().__init__()
        self.block0 = Conv2(in_channels, out_channels, global_cond_channels)
        self.block1 = Conv2(out_channels, out_channels, global_cond_channels)

    def forward(self, x, global_cond):
        return self.block1(self.block0(x, global_cond), global_cond)

class RNN4(nn.Module):
    def __init__(self, in_channels, out_channels, warmup_steps, global_cond_channels):
        super().__init__()
        self.gru = nn.GRU(in_channels + global_cond_channels, out_channels, batch_first=True)
        self.tconv = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=4)
        self.warmup_steps = warmup_steps

    def forward(self, x, global_cond):
        if global_cond is not None:
            global_cond = global_cond.unsqueeze(1).expand(-1, x.size(1), -1)
        x1, h_n = self.gru(torch.cat(filter_none([x, global_cond]), dim=2))
        y = self.tconv(x1[:, self.warmup_steps:].transpose(1, 2)).transpose(1, 2)
        return y, h_n.squeeze(0)

    def to_cell(self):
        return RNN4Cell(self.gru, self.tconv)

class RNN4Cell(nn.Module):
    def __init__(self, gru, tconv):
        super().__init__()

        self.gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        self.gru_cell.weight_hh.data = gru.weight_hh_l0.data
        self.gru_cell.weight_ih.data = gru.weight_ih_l0.data
        self.gru_cell.bias_hh.data = gru.bias_hh_l0.data
        self.gru_cell.bias_ih.data = gru.bias_ih_l0.data
        self.tconv = tconv

    def forward(self, x, global_cond, h):
        h1 = self.gru_cell(torch.cat(filter_none([x, global_cond]), dim=1), h)
        y = self.tconv(h1.unsqueeze(2)).transpose(1, 2)
        return y, h1

class Overtone(nn.Module):
    def __init__(self, wrnn_dims, fc_dims, cond_channels, global_cond_channels):
        super().__init__()
        conv_channels = 128
        rnn_channels = 512
        self.warmup_steps = 64
        self.conv0 = Conv4(1, conv_channels, global_cond_channels)
        self.conv1 = Conv4(conv_channels, conv_channels, global_cond_channels)
        self.conv2 = Conv4(conv_channels, conv_channels, global_cond_channels)
        self.rnn0 = RNN4(conv_channels + cond_channels, rnn_channels, self.warmup_steps, global_cond_channels)
        self.rnn1 = RNN4(conv_channels + rnn_channels, rnn_channels, self.warmup_steps, global_cond_channels)
        self.rnn2 = RNN4(conv_channels + rnn_channels, rnn_channels, self.warmup_steps, global_cond_channels)
        self.wavernn = WaveRNN(wrnn_dims, fc_dims, rnn_channels + global_cond_channels, 0)

        self.delay_c0 = 9
        self.delay_c1 = self.delay_c0 + 9 * 4
        self.delay_c2 = self.delay_c1 + 9 * 16
        self.delay_r0 = self.delay_c2 + self.warmup_steps * 64
        self.delay_r1 = self.delay_r0 + self.warmup_steps * 16
        self.delay_r2 = self.delay_r1 + self.warmup_steps * 4
        self.delay_wr = self.delay_r2 + self.warmup_steps

        cond_delay = self.delay_wr - self.delay_c2
        if cond_delay % 64 != 0:
            raise RuntimeError(f'Overtone: bad cond delay: {cond_delay}')
        self.cond_pad = cond_delay // 64

    def forward(self, x, cond, global_cond):
        n = x.size(0)
        x_coarse = x[:, :, :1]
        c0 = self.conv0(x_coarse, global_cond)
        c1 = self.conv1(c0, global_cond)
        c2 = self.conv2(c1, global_cond)
        r0 = self.rnn0(torch.cat(filter_none([c2, cond]), dim=2), global_cond)[0]
        r1 = self.rnn1(torch.cat([c1[:, (self.delay_r0 - self.delay_c1) // 16:], r0], dim=2), global_cond)[0]
        r2 = self.rnn2(torch.cat([c0[:, (self.delay_r1 - self.delay_c0) // 4:], r1], dim=2), global_cond)[0]
        if global_cond is not None:
            global_cond = global_cond.unsqueeze(1).expand(-1, r2.size(1), -1)
        cond_w = torch.cat(filter_none([r2, global_cond]), dim=2)
        p_c, p_f, _ = self.wavernn(x[:, self.delay_r2:], cond_w, None, None, None)
        return p_c[:, self.warmup_steps:], p_f[:, self.warmup_steps:]

    def generate(self, cond, global_cond, n=None, seq_len=None, verbose=False, use_half=False):
        start = time.time()
        if n is None:
            n = cond.size(0)
        if seq_len is None:
            seq_len = (cond.size(1) - self.cond_pad) * 64
        if use_half:
            std_tensor = torch.tensor([]).cuda().half()
        else:
            std_tensor = torch.tensor([]).cuda()

        # Warmup
        c0 = self.conv0(std_tensor.new_zeros(n, 10, 1), global_cond).repeat(1, 10, 1)
        c1 = self.conv1(c0, global_cond).repeat(1, 10, 1)
        c2 = self.conv2(c1, global_cond)

        if cond is None:
            pad_cond = None
        else:
            pad_cond = cond[:, :self.cond_pad]
        #logger.log(f'pad_cond: {pad_cond.size()}')
        r0, h0 = self.rnn0(torch.cat(filter_none([c2.repeat(1, 85, 1), pad_cond]), dim=2), global_cond)
        r1, h1 = self.rnn1(torch.cat([c1.repeat(1, 9, 1)[:, :84], r0], dim=2), global_cond)
        r2, h2 = self.rnn2(torch.cat([c0.repeat(1, 8, 1), r1], dim=2), global_cond)
        if global_cond is not None:
            global_cond_1 = global_cond.unsqueeze(1).expand(-1, r2.size(1), -1)
        else:
            global_cond_1 = None
        h3 = self.wavernn(std_tensor.new_zeros(n, 64, 3), torch.cat(filter_none([r2, global_cond_1]), dim=2))[2]

        # Create cells
        cell0 = self.rnn0.to_cell()
        cell1 = self.rnn1.to_cell()
        cell2 = self.rnn2.to_cell()
        wcell = self.wavernn.to_cell()

        # Main loop!
        coarse = std_tensor.new_zeros(n, 10, 1)
        c_val = std_tensor.new_zeros(n)
        f_val = std_tensor.new_zeros(n)
        zero = std_tensor.new_zeros(n)
        output = []
        for t in range(seq_len):
            #logger.log(f't = {t}')
            t0 = t % 4
            ct0 = (-t) % 4

            if t0 == 0:
                t1 = (t // 4) % 4
                ct1 = ((-t) // 4) % 4

                #logger.log(f'written to c0[{-ct1-1}]')
                c0[:, -ct1-1].copy_(self.conv0(coarse, global_cond).squeeze(1))
                coarse[:, :-4].copy_(coarse[:, 4:])

                if t1 == 0:
                    t2 = (t // 16) % 4
                    ct2 = ((-t) // 16) % 4

                    #logger.log('read c0')
                    #logger.log(f'written to c1[{-ct2-1}]')
                    c1[:, -ct2-1].copy_(self.conv1(c0, global_cond).squeeze(1))
                    c0[:, :-4].copy_(c0[:, 4:])

                    if t2 == 0:
                        #logger.log('read c1')
                        #logger.log('written to c2')
                        c2 = self.conv2(c1, global_cond).squeeze(1)
                        c1[:, :-4].copy_(c1[:, 4:])

                        #logger.log('read c2')
                        #logger.log('written to r0')
                        if cond is None:
                            inp0 = c2
                        else:
                            inp0 = torch.cat([c2, cond[:, t // 64 + self.cond_pad]], dim=1)
                        r0, h0 = cell0(inp0, global_cond, h0)

                    #logger.log(f'read r0[{t2}]')
                    #logger.log(f'written to r1')
                    #logger.log(f'c1: {c1.size()} r0: {r0.size()}')
                    r1, h1 = cell1(torch.cat([c1[:, -ct2-1], r0[:, t2]], dim=1), global_cond, h1)

                #logger.log(f'read r1[{t1}]')
                #logger.log(f'written to r2')
                #logger.log(f'c0: {c0.size()} r1: {r1.size()}')
                r2, h2 = cell2(torch.cat([c0[:, -ct1-1], r1[:, t1]], dim=1), global_cond, h2)

            #logger.log(f'read r2[{t0}]')
            wcond = torch.cat(filter_none([r2[:, t0], global_cond]), dim=1)

            x = torch.stack([c_val, f_val, zero], dim=1)
            o_c = wcell.forward_c(x, wcond, None, None, h3)
            c_cat = utils.nn.sample_softmax(o_c).float()
            c_val_new = (c_cat / 127.5 - 1.0).to(std_tensor)

            x = torch.stack([c_val, f_val, c_val_new], dim=1)
            o_f, h3 = wcell.forward_f(x, wcond, None, None, h3)
            f_cat = utils.nn.sample_softmax(o_f).float()
            f_val = (f_cat / 127.5 - 1.0).to(std_tensor)
            c_val = c_val_new

            sample = (c_cat * 256 + f_cat) / 32767.5 - 1.0
            coarse[:, 6+t0].copy_(c_val.unsqueeze(1))

            if verbose and t % 10000 < 100:
                logger.log(f'c={c_cat[0]} f={f_cat[0]} sample={sample[0]}')
            output.append(sample)
            if t % 100 == 0 :
                speed = int((t + 1) / (time.time() - start))
                logger.status(f'{t+1}/{seq_len} -- Speed: {speed} samples/sec')

        return torch.stack(output, dim=1)

    def after_update(self):
        self.wavernn.after_update()

    def pad(self):
        return self.delay_wr

def filter_none(xs):
    return [x for x in xs if x is not None]

