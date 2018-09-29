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
from layers.vector_quant import VectorQuant
from layers.downsampling_encoder import DownsamplingEncoder
import utils.env as env
import utils.logger as logger

class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, upsample_factors, normalize_vq=False):
        super().__init__()
        self.n_classes = 256
        self.upsample = UpsampleNetwork(upsample_factors, pad=1)
        self.wavernn = WaveRNN(rnn_dims, fc_dims, 128, 0)
        self.vq = VectorQuant(1, 512, 128, normalize=normalize_vq)
        encoder_layers = [
            (2, 4, False),
            (2, 4, False),
            (2, 4, False),
            (2, 4, False),
            (2, 4, False),
            (2, 4, False),
            (2, 4, True),
            (2, 4, True),
            (2, 4, True),
            (2, 4, True),
            ]
        self.encoder = DownsamplingEncoder(128, encoder_layers)
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

        cond = self.upsample(discrete.squeeze(2).transpose(1, 2))
        # cond: (N, 768, 64)
        #logger.log(f'cond: {cond.size()}')
        return self.wavernn(x, cond, None, None, None), vq_pen.mean(), encoder_pen.mean(), entropy

    def after_update(self):
        self.wavernn.after_update()

    def generate(self, samples, save_path, deterministic=False) :
        samples = torch.FloatTensor(samples).cuda()
        # samples: (L)
        #logger.log(f'samples: {samples.size()}')
        self.eval()
        with torch.no_grad() :
            continuous = self.encoder(samples.unsqueeze(0))
            discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
            logger.log(f'entropy: {entropy}')
            cond = self.upsample(discrete.squeeze(2).transpose(1, 2))
            # cond: (1, L1, 64)
            #logger.log(f'cond: {cond.size()}')
            output = self.wavernn.generate(cond, None, None, None)
        librosa.output.write_wav(save_path, output, sample_rate)
        self.train()
        return output

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logger.log('Trainable Parameters: %.3f million' % parameters)

    def pad_left(self):
        return self.encoder.pad_left + self.encoder.total_scale

    def pad_right(self):
        return self.encoder.total_scale

    def total_scale(self):
        return self.encoder.total_scale

def train(paths, model, dataset, optimiser, epochs, batch_size, seq_len, step, lr=1e-4) :

    optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
    for p in optimiser.param_groups : p['lr'] = lr
    criterion = nn.NLLLoss().cuda()
    k = 0
    saved_k = 0
    pad_left = model.pad_left()
    pad_right = model.pad_right()
    time_span = pad_left + pad_right + 16 * model.total_scale()
    logger.log(f'pad_left={pad_left}, pad_right={pad_right}, total_scale={model.total_scale()}')

    for e in range(epochs) :

        trn_loader = DataLoader(dataset, collate_fn=lambda batch: env.collate_samples(time_span, batch), batch_size=batch_size,
                                num_workers=2, shuffle=True, pin_memory=True)

        start = time.time()
        running_loss_c = 0.
        running_loss_f = 0.
        running_loss_vq = 0.
        running_entropy = 0.

        iters = len(trn_loader)

        for i, (coarse, fine, coarse_f, fine_f) in enumerate(trn_loader) :

            coarse, fine, coarse_f, fine_f = coarse.cuda(), fine.cuda(), coarse_f.cuda().half(), fine_f.cuda().half()
            #coarse, fine, coarse_f, fine_f = coarse.cuda(), fine.cuda(), coarse_f.cuda(), fine_f.cuda()

            x = torch.cat([
                coarse_f[:, pad_left-1:-pad_right-1].unsqueeze(-1),
                fine_f[:, pad_left-1:-pad_right-1].unsqueeze(-1),
                coarse_f[:, pad_left:-pad_right].unsqueeze(-1),
                ], dim=2)
            y_coarse = coarse[:, pad_left:-pad_right]
            y_fine = fine[:, pad_left:-pad_right]

            p_cf, vq_pen, encoder_pen, entropy = model(x, coarse_f)
            p_c, p_f = p_cf
            loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
            loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
            loss = loss_c + loss_f + vq_pen + 0.01 * encoder_pen

            optimiser.zero_grad()
            #loss.backward()
            optimiser.backward(loss)
            optimiser.step()
            running_loss_c += loss_c.item()
            running_loss_f += loss_f.item()
            running_loss_vq += vq_pen.item()
            running_entropy += entropy

            model.after_update()

            speed = (i + 1) / (time.time() - start)
            avg_loss_c = running_loss_c / (i + 1)
            avg_loss_f = running_loss_f / (i + 1)
            avg_loss_vq = running_loss_vq / (i + 1)
            avg_entropy = running_entropy / (i + 1)

            step += 1
            k = step // 1000
            logger.status(f'Epoch: {e+1}/{epochs} -- Batch: {i+1}/{iters} -- Loss: c={avg_loss_c:#.4} f={avg_loss_f:#.4} vq={avg_loss_vq:#.4} -- Entropy: {avg_entropy:#.4} -- Speed: {speed:#.4} steps/sec -- Step: {k}k ')

        torch.save(model.state_dict(), paths.model_path())
        np.save(paths.step_path(), step)
        logger.log_current_status()
        logger.log(f' <saved>; w[0][0] = {model.wavernn.gru.weight_ih_l0[0][0]}')
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
        logger.log('Generating: %i/%i' % (i+1, samples))
        gt = 2 * gt.astype(np.float32) / (2**env.bits - 1.) - 1.
        librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', gt, sr=sample_rate)
        output = model.generate(gt, f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', deterministic)
