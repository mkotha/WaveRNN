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
import argparse
import platform

parser = argparse.ArgumentParser(description='Train or run some neural net')
parser.add_argument('--generate', '-g', action='store_true')
parser.add_argument('--float', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load', '-l')
parser.add_argument('--scratch', action='store_true')
args = parser.parse_args()

if args.float and args.half:
    sys.exit('--float and --half cannot be specified together')

if args.float:
    use_half = False
elif args.half:
    use_half = True
else:
    use_half = not args.generate

seq_len = hop_length * 5

model_name = 'vq.9.wide'

if platform.node().endswith('.ec2') or platform.node().startswith('ip-'): # Running on EC2
    DATA_PATH = '/home/ubuntu/dataset/lj-16bit'
else:
    DATA_PATH = '/mnt/backup/dataset/lj-16bit'

paths = env.Paths(model_name, DATA_PATH)

with open(f'{DATA_PATH}/dataset_ids.pkl', 'rb') as f:
    dataset_ids = pickle.load(f)

#test_ids = dataset_ids[-50:]
#dataset_ids = dataset_ids[:-50]
test_ids = dataset_ids[-3:]
dataset_ids = dataset_ids[:-3]


dataset = env.AudiobookDataset(dataset_ids, DATA_PATH)

print(f'dataset size: {len(dataset)}')

model = vqvae.Model(rnn_dims=896, fc_dims=896,
              upsample_factors=(4, 4, 4), normalize_vq=True).cuda()

if use_half:
    model = model.half()

if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
    # Start from scratch
    step = 0
else:
    if args.load:
        prev_path = args.load
    else:
        prev_path = paths.model_path()
    step = env.restore(prev_path, model)

optimiser = optim.Adam(model.parameters())

if args.generate:
    vqvae.generate(paths, model, step, DATA_PATH, test_ids)#, deterministic=True)
else:
    vqvae.train(paths, model, dataset, optimiser, epochs=1000, batch_size=16, seq_len=seq_len, step=step, lr=1e-4)

