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



class ResBlock(nn.Module) :
    def __init__(self, dims) :
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x) :
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


# In[18]:


class MelResNet(nn.Module) :
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims) :
        super().__init__()
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=5, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks) :
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x) :
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers : x = f(x)
        x = self.conv_out(x)
        return x


# In[16]:


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


# In[19]:


class UpsampleNetwork(nn.Module) :
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad) :
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims)
        self.resnet_stretch = Stretch2d(total_scale, 1)
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
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers : m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


# In[20]:


class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks):
        super().__init__()
        self.n_classes = 256
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 3
        self.half_rnn_dims = rnn_dims // 2
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims,
                                        res_blocks, res_out_dims, pad)
        self.rnn = nn.GRU(feat_dims + self.aux_dims + 3, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(self.half_rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, self.n_classes)
        self.fc3 = nn.Linear(self.half_rnn_dims + self.aux_dims, fc_dims)
        self.fc4 = nn.Linear(fc_dims, self.n_classes)

        coarse_mask = torch.cat([torch.ones(self.half_rnn_dims, feat_dims + self.aux_dims + 2), torch.zeros(self.half_rnn_dims, 1)], dim=1)
        i2h_mask = torch.cat([coarse_mask, torch.ones(self.half_rnn_dims, feat_dims + self.aux_dims + 3)], dim=0)
        self.mask = torch.cat([i2h_mask, i2h_mask, i2h_mask], dim=0).cuda()
        self.num_params()

    def forward(self, x, mels) :
        #print(f'x: {x.var()} mels: {mels.var()}')
        mels, aux = self.upsample(mels)

        #print(f'aux: {aux.var()} mels: {mels.var()}')
        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]

        #print(f'mels: {mels.size()}, a1: {a1.size()}, x: {x.size()}')
        x = torch.cat([mels, a1, x], dim=2)
        h, _ = self.rnn(x)
        #print(f'h: {h.var()}')

        h_c, h_f = torch.split(h, self.half_rnn_dims, dim=2)

        o_c = F.relu(self.fc1(torch.cat([h_c, a2], dim=2)))
        p_c = F.log_softmax(self.fc2(o_c), dim=2)
        #print(f'o_c: {o_c.var()} p_c: {p_c.var()}')

        o_f = F.relu(self.fc3(torch.cat([h_f, a3], dim=2)))
        p_f = F.log_softmax(self.fc4(o_f), dim=2)
        #print(f'o_f: {o_f.var()} p_f: {p_f.var()}')

        return (p_c, p_f)

    def after_update(self):
        with torch.no_grad():
            torch.mul(self.rnn.weight_ih_l0.data, self.mask, out=self.rnn.weight_ih_l0.data)

    def preview_upsampling(self, mels) :
        mels, aux = self.upsample(mels)
        return mels, aux

    def generate(self, mels, save_path, deterministic=False) :
        self.eval()
        output = []
        rnn_cell = self.get_gru_cell(self.rnn)
        with torch.no_grad() :
            start = time.time()
            h = torch.zeros(1, self.rnn_dims).cuda()

            mels = torch.FloatTensor(mels).cuda().unsqueeze(0)
            mels, aux = self.upsample(mels)

            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]

            seq_len = mels.size(1)

            c_val = 0.0
            f_val = 0.0

            for i in range(seq_len) :
                m_t = mels[:, i, :]
                a1_t = a1[:, i, :]
                a2_t = a2[:, i, :]
                a3_t = a3[:, i, :]

                x = torch.FloatTensor([[c_val, f_val, 0]]).cuda()

                x = torch.cat([m_t, a1_t, x], dim=1)
                h_0 = rnn_cell(x, h)

                h_c, _ = torch.split(h_0, self.half_rnn_dims, dim=1)

                o_c = F.relu(self.fc1(torch.cat([h_c, a2_t], dim=1)))
                if deterministic:
                    c_cat = torch.argmax(self.fc2(o_c), dim=1).to(torch.float32)[0]
                else:
                    posterior_c = F.softmax(self.fc2(o_c), dim=1)
                    distrib_c = torch.distributions.Categorical(posterior_c)
                    c_cat = distrib_c.sample().float().item()
                c_val_new = c_cat / 127.5 - 1.0

                x = torch.FloatTensor([[c_val, f_val, c_val_new]]).cuda()

                x = torch.cat([m_t, a1_t, x], dim=1)
                h = rnn_cell(x, h)

                _, h_f = torch.split(h, self.half_rnn_dims, dim=1)

                o_f = F.relu(self.fc3(torch.cat([h_f, a3_t], dim=1)))
                if deterministic:
                    f_cat = torch.argmax(self.fc4(o_f), dim=1).to(torch.float32)[0]
                else:
                    posterior_f = F.softmax(self.fc4(o_f), dim=1)
                    distrib_f = torch.distributions.Categorical(posterior_f)
                    f_cat = distrib_f.sample().float().item()
                f_val = f_cat / 127.5 - 1.0

                c_val = c_val_new

                sample = (c_cat * 256 + f_cat) / 32767.5 - 1.0
                if i % 10000 < 100:
                    print(f'c={c_cat} f={f_cat} sample={sample}')
                output.append(sample)
                if i % 100 == 0 :
                    speed = int((i + 1) / (time.time() - start))
                    print(f'\r{i+1}/{seq_len} -- Speed: {speed} samples/sec', end='')
        output = np.array(output).astype(np.float32)
        librosa.output.write_wav(save_path, output, sample_rate)
        self.train()
        return output

    def get_gru_cell(self, gru) :
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)


def train(paths, model, dataset, optimiser, epochs, batch_size, seq_len, step, lr=1e-4) :

    optimiser = apex.fp16_utils.FP16_Optimizer(optimiser)
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

            x, m, y_coarse, y_fine = x.cuda(), m.cuda(), y_coarse.cuda(), y_fine.cuda()

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
        print(f'\n <saved>; w[0][0] = {model.rnn.weight_ih_l0[0][0]}')
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

def try_restore(paths, model):
    if not os.path.exists(paths.model_path()):
        torch.save(model.state_dict(), paths.model_path())
    model.load_state_dict(torch.load(paths.model_path()))

    if not os.path.exists(paths.step_path()):
        np.save(paths.step_path(), 0)
    return np.load(paths.step_path())
