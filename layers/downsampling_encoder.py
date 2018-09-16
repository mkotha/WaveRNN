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
        #print(f'sd[samples] {x.std()}')
        for i, conv in enumerate(self.convs):
            x = conv(x)
            #print(f'sd[conv] {x.std()}')
            if i < len(self.convs) - 1:
                x = F.relu(x)
                #print(f'sd[relu] {x.std()}')
        return x.transpose(1, 2)
