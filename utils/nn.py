import torch
import torch.nn.functional as F

def sample_softmax(score):
    """ Sample from the softmax distribution represented by scores.

    Input:
        score: (N, D) numeric tensor
    Output:
        sample: (N) long tensor, 0 <= sample < D
    """
    posterior = F.softmax(score, dim=1)
    return torch.distributions.Categorical(posterior).sample()
