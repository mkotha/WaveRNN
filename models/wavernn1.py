import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from utils.dsp import *
import sys
import time
import apex
from layers.wavernn import WaveRNN
from layers.upsample import UpsampleNetwork
import utils.env as env

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

    def load_state_dict(self, dict):
        return super().load_state_dict(upgrade_state_dict(dict))

def upgrade_state_dict(state_dict):
    out_dict = {}
    for key, val in state_dict.items():
        if key in UPGRADE_KEY:
            key = UPGRADE_KEY[key]
        out_dict[key] = val
    return out_dict

def train(paths, model, dataset, optimiser, epochs, batch_size, seq_len, step, lr=1e-4) :

    optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
    for p in optimiser.param_groups : p['lr'] = lr
    criterion = nn.NLLLoss().cuda()
    k = 0
    saved_k = 0

    for e in range(epochs) :

        trn_loader = DataLoader(dataset, collate_fn=env.collate, batch_size=batch_size,
                                num_workers=2, shuffle=True, pin_memory=True)

        start = time.time()
        running_loss_c = 0.
        running_loss_f = 0.

        iters = len(trn_loader)

        for i, (x, m, y_coarse, y_fine) in enumerate(trn_loader) :

            x, m, y_coarse, y_fine = x.cuda().half(), m.cuda().half(), y_coarse.cuda(), y_fine.cuda()

            p_c, p_f = model(x, m)
            loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
            loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
            loss = loss_c + loss_f

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
        gt = 2 * gt.astype(np.float32) / (2**env.bits - 1.) - 1.
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
