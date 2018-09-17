import torch
import torch.nn as nn
import numpy as np

class UpsampleNetwork(nn.Module) :
    """
    Input: (N, C, L) numeric tensor

    Output: (N, C, L1) numeric tensor
    """
    def __init__(self, feat_dims, upsample_scales):
        super().__init__()
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            if scale % 2 == 1:
                ksz = 2 * scale + 1
                padding = (3 * scale - 1) // 2
            else:
                ksz = 2 * scale
                padding = 3 * scale // 2 - 1
            conv = nn.ConvTranspose2d(1, 1,
                    kernel_size = (1, ksz),
                    stride = (1, scale),
                    padding = (0, padding))
            if scale % 2 == 1:
                conv.weight.data.copy_(1 - ((torch.arange(ksz).float() - scale) / scale).abs())
            else:
                conv.weight.data.copy_(1 - ((torch.arange(ksz).float() * 2 + 1 - 2 * scale) / (2 * scale)).abs())
            conv.bias.data.zero_()
            self.up_layers.append(conv)

    def forward(self, mels):
        x = mels.unsqueeze(1)
        for up in self.up_layers:
            x = up(x)
        return x.squeeze(1)[:, :, 1:-1]
