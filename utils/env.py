from torch.utils.data import Dataset
import torch
import os
import numpy as np
from utils.dsp import *
import re

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

    def logfile_path(self):
        return f'log/{self.name}'

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

def collate_samples(left_pad, window, right_pad, batch):
    #print(f'collate: window={window}')
    samples = [x[1] for x in batch]
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    wave16 = np.stack(wave16).astype(np.int64) + 2**15
    coarse = wave16 // 256
    fine = wave16 % 256

    coarse = torch.LongTensor(coarse)
    fine = torch.LongTensor(fine)

    coarse_f = coarse.float() / 127.5 - 1.
    fine_f = fine.float() / 127.5 - 1.

    return coarse, fine, coarse_f, fine_f

def collate(left_pad, mel_win, right_pad, batch) :
    max_offsets = [x[0].shape[-1] - mel_win for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [offset * hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x[1], np.zeros(right_pad, dtype=np.int16)])[sig_offsets[i]:sig_offsets[i] + left_pad + 64 * mel_win + right_pad] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    wave16 = np.stack(wave16).astype(np.int64) + 2**15
    coarse = wave16 // 256
    fine = wave16 % 256

    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    fine = torch.LongTensor(fine)

    coarse_f = coarse.float() / 127.5 - 1.
    fine_f = fine.float() / 127.5 - 1.

    return mels, coarse, fine, coarse_f, fine_f

def restore(path, model):
    model.load_state_dict(torch.load(path))

    match = re.search(r'_([0-9]+)\.pyt', path)
    if match:
        return int(match.group(1))

    step_path = re.sub(r'\.pyt', '_step.npy', path)
    return np.load(step_path)
