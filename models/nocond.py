import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.dsp import *
import sys
import time
from layers.overtone import Overtone
from layers.upsample import UpsampleNetwork
from layers.vector_quant import VectorQuant
from layers.downsampling_encoder import DownsamplingEncoder
import utils.env as env
import utils.logger as logger

class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims):
        super().__init__()
        self.n_classes = 256
        self.overtone = Overtone(rnn_dims, fc_dims, 0, 0)
        self.num_params()

    def forward(self, x):
        p_c, p_f = self.overtone(x, None, None)
        return p_c, p_f

    def after_update(self):
        self.overtone.after_update()

    def generate(self, batch_size, seq_len, deterministic=False):
        self.eval()
        with torch.no_grad() :
            output = self.overtone.generate(None, None, seq_len=seq_len, n=batch_size)
        self.train()
        return output

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def do_train(self, paths, dataset, optimiser, epochs, batch_size, step, lr=1e-4, valid_index=[], use_half=False):

        if use_half:
            import apex
            optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
        for p in optimiser.param_groups : p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        k = 0
        saved_k = 0
        pad_left = self.overtone.pad()
        time_span = 16 * 64

        for e in range(epochs) :

            trn_loader = DataLoader(dataset, collate_fn=lambda batch: env.collate_samples(pad_left, time_span, 1, batch), batch_size=batch_size,
                                    num_workers=2, shuffle=True, pin_memory=True)

            start = time.time()
            running_loss_c = 0.
            running_loss_f = 0.
            max_grad = 0.
            max_grad_name = ""

            iters = len(trn_loader)

            for i, wave16 in enumerate(trn_loader) :

                wave16 = wave16.cuda()

                coarse = (wave16 + 2**15) // 256
                fine = (wave16 + 2**15) % 256

                coarse_f = coarse.float() / 127.5 - 1.
                fine_f = fine.float() / 127.5 - 1.

                if use_half:
                    coarse_f = coarse_f.half()
                    fine_f = fine_f.half()

                x = torch.cat([
                    coarse_f[:, :-1].unsqueeze(-1),
                    fine_f[:, :-1].unsqueeze(-1),
                    coarse_f[:, 1:].unsqueeze(-1),
                    ], dim=2)
                y_coarse = coarse[:, pad_left+1:]
                y_fine = fine[:, pad_left+1:]

                p_c, p_f = self(x)
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                loss = loss_c + loss_f

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                else:
                    loss.backward()
                    for name, param in self.named_parameters():
                        param_max_grad = param.grad.data.abs().max()
                        if param_max_grad > max_grad:
                            max_grad = param_max_grad
                            max_grad_name = name
                    nn.utils.clip_grad_norm_(self.parameters(), 1, 'inf')
                optimiser.step()
                running_loss_c += loss_c.item()
                running_loss_f += loss_f.item()

                self.after_update()

                speed = (i + 1) / (time.time() - start)
                avg_loss_c = running_loss_c / (i + 1)
                avg_loss_f = running_loss_f / (i + 1)

                step += 1
                k = step // 1000
                logger.status(f'Epoch: {e+1}/{epochs} -- Batch: {i+1}/{iters} -- Loss: c={avg_loss_c:#.4} f={avg_loss_f:#.4} -- Grad: {max_grad:#.1} {max_grad_name} Speed: {speed:#.4} steps/sec -- Step: {k}k ')

            os.makedirs(paths.checkpoint_dir, exist_ok=True)
            torch.save(self.state_dict(), paths.model_path())
            np.save(paths.step_path(), step)
            logger.log_current_status()
            logger.log(f' <saved>; w[0][0] = {self.overtone.wavernn.gru.weight_ih_l0[0][0]}')
            if k > saved_k + 50:
                torch.save(self.state_dict(), paths.model_hist_path(step))
                saved_k = k
                self.do_generate(paths, step, dataset.path, valid_index)
                logger.log('done generation')

    def do_generate(self, paths, step, data_path, test_index, deterministic=False, use_half=False, verbose=False):
        out = self.generate(len(test_index), 100000)
        k = step // 1000
        os.makedirs(paths.gen_path(), exist_ok=True)
        for i in range(len(test_index)) :
            audio = out[i].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
