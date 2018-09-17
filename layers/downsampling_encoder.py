import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class DownsamplingEncoder(nn.Module):
    """
        Input: (N, samples_i) numeric tensor
        Output: (N, samples_o, channels) numeric tensor
    """
    def __init__(self, channels, layer_specs):
        super().__init__()

        self.convs_strided = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.skips = []
        prev_channels = 1
        for scale, ksz in layer_specs:
            conv_strided = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=scale)
            wsize = 2.967 / math.sqrt(ksz * prev_channels)
            conv_strided.weight.data.uniform_(-wsize, wsize)
            conv_strided.bias.data.zero_()
            self.convs_strided.append(conv_strided)

            conv_1x1 = nn.Conv1d(channels, channels, 1)
            conv_1x1.bias.data.zero_()
            self.convs_1x1.append(conv_1x1)

            self.skips.append(ksz - scale)

            prev_channels = channels

    def forward(self, samples):
        x = samples.unsqueeze(1)
        #print(f'sd[samples] {x.std()}')
        for i, stuff in enumerate(zip(self.convs_strided, self.convs_1x1, self.skips)):
            conv_strided, conv_1x1, skip = stuff

            x1 = conv_strided(x)
            #print(f'sd[conv.s] {x1.std()}')
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            #print(f'sd[act] {x2.std()}')
            x3 = conv_1x1(x2)
            #print(f'sd[conv.1] {x3.std()}')
            if i == 0:
                x = x3
            else:
                x = x3 + x[:, :, skip:].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
            #print(f'sd[out] {x.std()}')
        return x.transpose(1, 2)
