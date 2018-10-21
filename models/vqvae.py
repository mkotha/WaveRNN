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
import apex
from layers.overtone import Overtone
from layers.vector_quant import VectorQuant
from layers.downsampling_encoder import DownsamplingEncoder
import utils.env as env
import utils.logger as logger
import random

class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, upsample_factors, normalize_vq=False):
        super().__init__()
        self.n_classes = 256
        self.overtone = Overtone(rnn_dims, fc_dims, 128)
        self.vq = VectorQuant(1, 512, 128, normalize=normalize_vq)
        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(128, encoder_layers)
        self.frame_advantage = 4
        self.num_params()

    def forward(self, x, samples):
        # x: (N, 768, 3)
        #logger.log(f'x: {x.size()}')
        # samples: (N, 1022)
        #logger.log(f'samples: {samples.size()}')
        continuous = self.encoder(samples)
        # continuous: (N, 14, 64)
        #logger.log(f'continuous: {continuous.size()}')
        discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
        # discrete: (N, 14, 1, 64)
        #logger.log(f'discrete: {discrete.size()}')

        # cond: (N, 768, 64)
        #logger.log(f'cond: {cond.size()}')
        return self.overtone(x, discrete.squeeze(2)), vq_pen.mean(), encoder_pen.mean(), entropy

    def after_update(self):
        self.overtone.after_update()
        self.vq.after_update()

    def forward_generate(self, samples, deterministic=False, use_half=False, verbose=False):
        if use_half:
            samples = samples.half()
        # samples: (L)
        #logger.log(f'samples: {samples.size()}')
        self.eval()
        with torch.no_grad() :
            continuous = self.encoder(samples)
            discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
            logger.log(f'entropy: {entropy}')
            # cond: (1, L1, 64)
            #logger.log(f'cond: {cond.size()}')
            output = self.overtone.generate(discrete.squeeze(2), use_half=use_half, verbose=verbose)
        self.train()
        return output

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logger.log('Trainable Parameters: %.3f million' % parameters)

    def load_state_dict(self, dict):
        return super().load_state_dict(self.upgrade_state_dict(dict))

    def upgrade_state_dict(self, state_dict):
        out_dict = state_dict.copy()
        return out_dict

    def pad_left(self):
        return max(self.pad_left_decoder(), self.pad_left_encoder())

    def pad_left_decoder(self):
        return self.overtone.pad()

    def pad_left_encoder(self):
        return self.encoder.pad_left - self.frame_advantage * self.encoder.total_scale

    def pad_right(self):
        return self.frame_advantage * self.encoder.total_scale

    def total_scale(self):
        return self.encoder.total_scale

    def do_train(self, paths, dataset, optimiser, epochs, batch_size, seq_len, step, lr=1e-4, valid_ids=[], use_half=False):

        if use_half:
            optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
        for p in optimiser.param_groups : p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        k = 0
        saved_k = 0
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        extra_pad_right = 127
        pad_right = self.pad_right() + extra_pad_right
        window = 16 * self.total_scale()
        logger.log(f'pad_left={pad_left_encoder}|{pad_left_decoder}, pad_right={pad_right}, total_scale={self.total_scale()}')

        for e in range(epochs) :

            trn_loader = DataLoader(dataset, collate_fn=lambda batch: env.collate_samples(pad_left, window, pad_right, batch), batch_size=batch_size,
                                    num_workers=2, shuffle=True, pin_memory=True)

            start = time.time()
            running_loss_c = 0.
            running_loss_f = 0.
            running_loss_vq = 0.
            running_loss_vqc = 0.
            running_entropy = 0.
            max_grad = 0.
            max_grad_name = ""

            iters = len(trn_loader)

            for i, (coarse, fine, coarse_f, fine_f) in enumerate(trn_loader) :

                coarse, fine, coarse_f, fine_f = coarse.cuda(), fine.cuda(), coarse_f.cuda(), fine_f.cuda()
                if use_half:
                    coarse_f = coarse_f.half()
                    fine_f = fine_f.half()

                x = torch.cat([
                    coarse_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    fine_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    coarse_f[:, pad_left-pad_left_decoder+1:1-pad_right].unsqueeze(-1),
                    ], dim=2)
                y_coarse = coarse[:, pad_left+1:1-pad_right]
                y_fine = fine[:, pad_left+1:1-pad_right]

                # Randomly translate the input to the encoder to encourage
                # translational invariance
                total_len = coarse_f.size(1)
                translated = []
                for j in range(coarse_f.size(0)):
                    shift = random.randrange(256) - 128
                    translated.append(coarse_f[j, pad_left-pad_left_encoder+shift:total_len-extra_pad_right+shift])
                p_cf, vq_pen, encoder_pen, entropy = self(x, torch.stack(translated, dim=0))
                p_c, p_f = p_cf
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                encoder_weight = 0.01 * min(1, max(0.1, step / 1000 - 1))
                loss = loss_c + loss_f + vq_pen + encoder_weight * encoder_pen

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                else:
                    loss.backward()
                    # Commenting out gradient clipping because it's very
                    # expensive.
                    #
                    #for name, param in self.named_parameters():
                    #    param_max_grad = param.grad.data.abs().max()
                    #    if param_max_grad > max_grad:
                    #        max_grad = param_max_grad
                    #        max_grad_name = name
                    #nn.utils.clip_grad_norm_(self.parameters(), 100, norm_type='inf')
                optimiser.step()
                running_loss_c += loss_c.item()
                running_loss_f += loss_f.item()
                running_loss_vq += vq_pen.item()
                running_loss_vqc += encoder_pen.item()
                running_entropy += entropy

                self.after_update()

                speed = (i + 1) / (time.time() - start)
                avg_loss_c = running_loss_c / (i + 1)
                avg_loss_f = running_loss_f / (i + 1)
                avg_loss_vq = running_loss_vq / (i + 1)
                avg_loss_vqc = running_loss_vqc / (i + 1)
                avg_entropy = running_entropy / (i + 1)

                step += 1
                k = step // 1000
                logger.status(f'Epoch: {e+1}/{epochs} -- Batch: {i+1}/{iters} -- Loss: c={avg_loss_c:#.4} f={avg_loss_f:#.4} vq={avg_loss_vq:#.4} vqc={avg_loss_vqc:#.4} -- Entropy: {avg_entropy:#.4} -- Grad: {max_grad:#.1} {max_grad_name} Speed: {speed:#.4} steps/sec -- Step: {k}k ')

            torch.save(self.state_dict(), paths.model_path())
            np.save(paths.step_path(), step)
            logger.log_current_status()
            logger.log(f' <saved>; w[0][0] = {self.overtone.wavernn.gru.weight_ih_l0[0][0]}')
            if k > saved_k + 50:
                torch.save(self.state_dict(), paths.model_hist_path(step))
                saved_k = k
                self.do_generate(paths, step, dataset.path, valid_ids)

    def do_generate(self, paths, step, data_path, test_ids, deterministic=False, use_half=False, verbose=False):
        k = step // 1000
        gt = [np.load(f'{data_path}/quant/{id}.npy') for id in test_ids]
        coarse = [((x.astype(np.int64) + 2**15) // 256).astype(np.float32) / 127.5 - 1.0 for x in gt]
        gt = [(x.astype(np.float32) + 0.5) / (2**15 - 0.5) for x in gt]
        extended = [np.concatenate([np.zeros(self.pad_left_encoder(), dtype=np.float32), x, np.zeros(self.pad_right(), dtype=np.float32)]) for x in coarse]
        maxlen = max([len(x) for x in extended])
        aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen-len(x))]) for x in extended]
        os.makedirs(paths.gen_path(), exist_ok=True)
        out = self.forward_generate(torch.stack(aligned).cuda(), verbose=verbose)
        logger.log(f'out: {out.size()}')
        for i, x in enumerate(gt) :
            audio = out[i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x, sr=sample_rate)
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
