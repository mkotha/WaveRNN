import torch.nn as nn
import torch
import torch.nn.functional as F

class DownsamplingEncoder(nn.Module):
    """
        Input: (N, samples_i) numeric tensor
        Output: (N, samples_o, channels) numeric tensor
    """
    def __init__(self, layer_specs):
        super().__init__()

        self.convs = nn.ModuleList()
        channels = 1
        for scale, ksz, nch in layer_specs:
            self.convs.append(nn.Conv1d(channels, nch, ksz, stride=scale))
            channels = nch

    def forward(self, samples):
        x = samples.unsqueeze(1)
        for conv in self.convs:
            x = F.relu(conv(x))
        return x.transpose(1, 2)
