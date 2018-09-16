import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
from utils.dsp import *
import sys
import time
import apex
from layers.wavernn import WaveRNN


bits = 16

seq_len = hop_length * 5

class Paths:
    def __init__(self, name, data_dir, checkpoint_dir="model_checkpoints", output_dir="model_outputs"):
        self.name = name
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

    def model_path(self):
        return f'{self.checkpoint_dir}/{self.name}.pyt'

    def model_hist_path(self, step):
        return f'{self.checkpoint_dir}/{self.name}_{step}.pyt'

    def step_path(self):
        return f'{self.checkpoint_dir}/{self.name}_step.npy'

    def gen_path(self):
        return f'{self.output_dir}/{self.name}/'

def default_paths(name, data_dir):
    return Paths(name, data_dir, checkpoint_dir="model_checkpoints", output_dir="model_outputs")

class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f'{self.path}/mel/{file}.npy')
        x = np.load(f'{self.path}/quant/{file}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)


# In[10]:


def collate(batch) :

    pad = 2
    mel_win = seq_len // hop_length + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win]             for i, x in enumerate(batch)]

    wave16 = [x[1][sig_offsets[i]:sig_offsets[i] + seq_len + 1]               for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    wave16 = np.stack(wave16).astype(np.int64)
    coarse = wave16 // 256
    fine = wave16 % 256

    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    fine = torch.LongTensor(fine)

    coarse_f = coarse.float() / 127.5 - 1.
    fine_f = fine.float() / 127.5 - 1.

    x = torch.cat([coarse_f[:, :-1].unsqueeze(-1), fine_f[:, :-1].unsqueeze(-1), coarse_f[:, 1:].unsqueeze(-1)], dim=2)

    return x, mels, coarse[:, 1:], fine[:, 1:]



class Stretch2d(nn.Module) :
    def __init__(self, x_scale, y_scale) :
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x) :
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)

class UpsampleNetwork(nn.Module) :
    def __init__(self, upsample_scales, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales :
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m) :
        m = m.unsqueeze(1)
        for f in self.up_layers : m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2)


# In[20]:


class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, pad, upsample_factors,
                 feat_dims):
        super().__init__()
        self.n_classes = 256
        self.upsample = UpsampleNetwork(upsample_factors, pad)
        self.wavernn = WaveRNN(rnn_dims, fc_dims, feat_dims, 0)
        self.num_params()

    def forward(self, x, mels) :
        #print(f'x: {x.size()} mels: {mels.size()}')
        cond = self.upsample(mels)
        #print(f'cond: {cond.size()}')
        return self.wavernn(x, cond, None, None, None)

    def after_update(self):
        self.wavernn.after_update()

    def preview_upsampling(self, mels) :
        return self.upsample(mels)

    def generate(self, mels, save_path, deterministic=False) :
        self.eval()
        with torch.no_grad() :
            mels = torch.FloatTensor(mels).cuda().unsqueeze(0)
            cond = self.upsample(mels)
            output = self.wavernn.generate(cond, None, None, None)
        librosa.output.write_wav(save_path, output, sample_rate)
        self.train()
        return output

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)


def train(paths, model, dataset, optimiser, epochs, batch_size, seq_len, step, lr=1e-4) :

    optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
    for p in optimiser.param_groups : p['lr'] = lr
    criterion = nn.NLLLoss().cuda()
    k = 0
    saved_k = 0

    for e in range(epochs) :

        trn_loader = DataLoader(dataset, collate_fn=collate, batch_size=batch_size,
                                num_workers=2, shuffle=True, pin_memory=True)

        start = time.time()
        running_loss_c = 0.
        running_loss_f = 0.

        iters = len(trn_loader)

        for i, (x, m, y_coarse, y_fine) in enumerate(trn_loader) :

            x, m, y_coarse, y_fine = x.cuda().half(), m.cuda().half(), y_coarse.cuda(), y_fine.cuda()

            p_c, p_f = model(x, m)
            #print(f'p_c: {p_c.size()}, p_f: {p_f.size()}, y_coarse: {y_coarse.size()}')
            #print(f'p_c: {p_c}')
            #print(f'p_f: {p_f}')
            loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
            loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
            loss = loss_c + loss_f
            #print(f'loss_c: {loss_c} loss_f: {loss_f} loss: {loss}')

            optimiser.zero_grad()
            #loss.backward()
            optimiser.backward(loss)
            optimiser.step()
            running_loss_c += loss_c.item()
            running_loss_f += loss_f.item()

            model.after_update()

            speed = (i + 1) / (time.time() - start)
            avg_loss_c = running_loss_c / (i + 1)
            avg_loss_f = running_loss_f / (i + 1)

            step += 1
            k = step // 1000
            print(f'\rEpoch: {e+1}/{epochs} -- Batch: {i+1}/{iters} -- Loss: c={avg_loss_c:#.4} f={avg_loss_f:#.4} -- Speed: {speed:#.4} steps/sec -- Step: {k}k ', end='')

        torch.save(model.state_dict(), paths.model_path())
        np.save(paths.step_path(), step)
        print(f'\n <saved>; w[0][0] = {model.wavernn.gru.weight_ih_l0[0][0]}')
        if k > saved_k + 50:
            torch.save(model.state_dict(), paths.model_hist_path(step))
            saved_k = k

def generate(paths, model, step, data_path, test_ids, samples=3, deterministic=False) :
    global output
    k = step // 1000
    test_mels = [np.load(f'{data_path}/mel/{id}.npy') for id in test_ids[:samples]]
    ground_truth = [np.load(f'{data_path}/quant/{id}.npy') for id in test_ids[:samples]]
    os.makedirs(paths.gen_path(), exist_ok=True)
    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)) :
        print('\nGenerating: %i/%i' % (i+1, samples))
        gt = 2 * gt.astype(np.float32) / (2**bits - 1.) - 1.
        librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', gt, sr=sample_rate)
        output = model.generate(mel, f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', deterministic)

UPGRADE_KEY = {
        "rnn.weight_ih_l0": "wavernn.gru.weight_ih_l0",
        "rnn.weight_hh_l0": "wavernn.gru.weight_hh_l0",
        "rnn.bias_ih_l0": "wavernn.gru.bias_ih_l0",
        "rnn.bias_hh_l0": "wavernn.gru.bias_hh_l0",
        "fc1.weight": "wavernn.fc1.weight",
        "fc1.bias": "wavernn.fc1.bias",
        "fc2.weight": "wavernn.fc2.weight",
        "fc2.bias": "wavernn.fc2.bias",
        "fc3.weight": "wavernn.fc3.weight",
        "fc3.bias": "wavernn.fc3.bias",
        "fc4.weight": "wavernn.fc4.weight",
        "fc4.bias": "wavernn.fc4.bias",
        }

def upgrade_state_dict(state_dict):
    out_dict = {}
    for key, val in state_dict.items():
        if key in UPGRADE_KEY:
            key = UPGRADE_KEY[key]
        out_dict[key] = val
    return out_dict

def try_restore(paths, model):
    if not os.path.exists(paths.model_path()):
        torch.save(model.state_dict(), paths.model_path())
    model.load_state_dict(upgrade_state_dict(torch.load(paths.model_path())))

    if not os.path.exists(paths.step_path()):
        np.save(paths.step_path(), 0)
    return np.load(paths.step_path())
