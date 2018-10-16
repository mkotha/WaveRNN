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
        if normalize:
            target_scale = 0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3
            self.normalize_scale = None
        self.embedding0 = nn.Parameter(torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.after_update()

    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
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
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy)

    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0.size(2))
                self.embedding0.mul_(target_norm / self.embedding0.norm(dim=2, keepdim=True))
