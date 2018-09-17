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
import utils.logger as logger

class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, pad, upsample_factors, feat_dims):
        super().__init__()
        self.n_classes = 256
        self.excess_pad = pad - 1
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors)
        self.wavernn = WaveRNN(rnn_dims, fc_dims, feat_dims, 0)
        self.num_params()

    def forward(self, x, mels) :
        #logger.log(f'x: {x.size()} mels: {mels.size()}')
        cond = self.upsample(mels[:, :, self.excess_pad:-self.excess_pad])
        #logger.log(f'cond: {cond.size()}')
        return self.wavernn(x, cond.transpose(1, 2), None, None, None)

    def after_update(self):
        self.wavernn.after_update()

    def preview_upsampling(self, mels) :
        return self.upsample(mels)

    def forward_generate(self, mels, deterministic=False, use_half=False, verbose=False):
        n = mels.size(0)
        if use_half:
            mels = mels.half()
        self.eval()
        with torch.no_grad() :
            cond = self.upsample(mels[:, :, self.excess_pad:-self.excess_pad])
            output = self.wavernn.generate(cond.transpose(1, 2), None, None, None, use_half=use_half, verbose=verbose)
        self.train()
        return output

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logger.log('Trainable Parameters: %.3f million' % parameters)

    def load_state_dict(self, dict):
        return super().load_state_dict(upgrade_state_dict(dict))

    def do_train(self, paths, dataset, optimiser, epochs, batch_size, seq_len, step, lr=1e-4, valid_ids=[], use_half=False):
        if use_half:
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

                x, m, y_coarse, y_fine = x.cuda(), m.cuda(), y_coarse.cuda(), y_fine.cuda()
                if use_half:
                    x = x.half()
                    m = m.half()

                p_c, p_f = self(x, m)
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                loss = loss_c + loss_f

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                else:
                    loss.backward()
                optimiser.step()
                running_loss_c += loss_c.item()
                running_loss_f += loss_f.item()

                self.after_update()

                speed = (i + 1) / (time.time() - start)
                avg_loss_c = running_loss_c / (i + 1)
                avg_loss_f = running_loss_f / (i + 1)

                step += 1
                k = step // 1000
                logger.status(f'Epoch: {e+1}/{epochs} -- Batch: {i+1}/{iters} -- Loss: c={avg_loss_c:#.4} f={avg_loss_f:#.4} -- Speed: {speed:#.4} steps/sec -- Step: {k}k ')

            torch.save(self.state_dict(), paths.model_path())
            np.save(paths.step_path(), step)
            logger.log_current_status()
            logger.log(f' <saved>; w[0][0] = {self.wavernn.gru.weight_ih_l0[0][0]}')
            if k > saved_k + 50:
                torch.save(self.state_dict(), paths.model_hist_path(step))
                saved_k = k
                self.do_generate(paths, step, dataset.path, valid_ids)

    def do_generate(self, paths, step, data_path, test_ids, deterministic=False, use_half=False, verbose=False):
        global output
        k = step // 1000
        test_mels = [np.load(f'{data_path}/mel/{id}.npy') for id in test_ids]
        maxlen = max([x.shape[1] for x in test_mels])
        aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(80, maxlen-x.shape[1]+1)], dim=1) for x in test_mels]
        out = self.forward_generate(torch.stack(aligned).cuda(), deterministic, use_half=use_half, verbose=verbose)

        os.makedirs(paths.gen_path(), exist_ok=True)
        for i, id in enumerate(test_ids):
            gt = np.load(f'{data_path}/quant/{id}.npy')
            gt = (gt.astype(np.float32) + 0.5) / (2**15 - 0.5)
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', gt, sr=sample_rate)
            audio = out[i][:len(gt)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)

def upgrade_state_dict(state_dict):
    out_dict = {}
    for key, val in state_dict.items():
        if key in UPGRADE_KEY:
            key = UPGRADE_KEY[key]
        out_dict[key] = val
    return out_dict

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
