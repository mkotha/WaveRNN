import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import utils.logger as logger

class VectorQuant(nn.Module):
    """
        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor
    """
    def __init__(self, n_channels, n_classes, vec_len, normalize=False):
        super().__init__()
        self.embedding0 = nn.Parameter(torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * 1e-3)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.normalize = normalize

    def forward(self, x0):
        if self.normalize:
            x = x0 / x0.norm(dim=3, keepdim=True)
            embedding = self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        index = (x1 - embedding).norm(dim=3).argmin(dim=2)
        # index: (N*samples, n_channels) long tensor
        if True: # compute the entropy
            hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
            #logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        index1 = (index + self.offset).view(index.size(0) * index.size(1))
        # index1: (N*samples*n_channels) long tensor
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2)
        return (out0, out1, out2, entropy)
