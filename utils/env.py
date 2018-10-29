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

class MultispeakerDataset(Dataset):
    def __init__(self, index, path):
        self.path = path
        self.index = index
        self.all_files = [(i, name) for (i, speaker) in enumerate(index) for name in speaker]

    def __getitem__(self, index):
        speaker_id, name = self.all_files[index]
        speaker_onehot = (np.arange(len(self.index)) == speaker_id).astype(np.long)
        audio = np.load(f'{self.path}/{speaker_id}/{name}.npy')
        return speaker_onehot, audio

    def __len__(self):
        return len(self.all_files)

    def num_speakers(self):
        return len(self.index)

def collate_multispeaker_samples(left_pad, window, right_pad, batch):
    samples = [x[1] for x in batch]
    speakers_onehot = torch.FloatTensor([x[0] for x in batch])
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.FloatTensor(speakers_onehot), torch.LongTensor(np.stack(wave16).astype(np.int64))

def collate_samples(left_pad, window, right_pad, batch):
    #print(f'collate: window={window}')
    samples = [x[1] for x in batch]
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.LongTensor(np.stack(wave16).astype(np.int64))

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

if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader
    DATA_PATH = 'vctk'
    with open(f'{DATA_PATH}/index.pkl', 'rb') as f:
        index = pickle.load(f)
    dataset = MultispeakerDataset(index, DATA_PATH)
    loader = DataLoader(dataset, batch_size=1)
    for x in loader:
        speaker_onehot, audio = x
        #print(f'x: {x}')
