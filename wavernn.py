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
import models.vqvae as vqvae
import utils.env as env


seq_len = hop_length * 5

#model_name = 'wavernn.0.sine'
#DATA_PATH = 'sinepp'

model_name = 'vq.0'
DATA_PATH = '/mnt/backup/dataset/lj-16bit'
#DATA_PATH = 'mepp16'

#paths = env.Paths(model_name, DATA_PATH, checkpoint_dir='remote-checkpoints/a')#, output_dir='deterministic')
paths = env.Paths(model_name, DATA_PATH)

#DATA_PATH = sys.argv[1]


with open(f'{DATA_PATH}/dataset_ids.pkl', 'rb') as f:
    dataset_ids = pickle.load(f)

#test_ids = dataset_ids[-50:]
#dataset_ids = dataset_ids[:-50]
test_ids = dataset_ids[-3:]
dataset_ids = dataset_ids[:-3]


dataset = env.AudiobookDataset(dataset_ids, DATA_PATH)

print(f'dataset size: {len(dataset)}')

model = vqvae.Model(rnn_dims=896, fc_dims=896,
              upsample_factors=(4, 4, 4)).cuda().half()


step = env.try_restore(paths, model)

optimiser = optim.Adam(model.parameters())


vqvae.train(paths, model, dataset, optimiser, epochs=1000, batch_size=16, seq_len=seq_len, step=step, lr=1e-4)

vqvae.generate(paths, model, step, DATA_PATH, test_ids)#, deterministic=True)
