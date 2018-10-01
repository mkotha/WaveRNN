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
import models.wavernn1 as wr
import utils.env as env
import argparse
import platform
import re
import utils.logger as logger
import time

parser = argparse.ArgumentParser(description='Train or run some neural net')
parser.add_argument('--generate', '-g', action='store_true')
parser.add_argument('--float', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load', '-l')
parser.add_argument('--scratch', action='store_true')
parser.add_argument('--model', '-m')
parser.add_argument('--force', action='store_true', help='skip the version check')
parser.add_argument('--count', '-c', type=int, help='number of audio files to generate')
args = parser.parse_args()

if args.float and args.half:
    sys.exit('--float and --half cannot be specified together')

if args.float:
    use_half = False
elif args.half:
    use_half = True
else:
    use_half = False

seq_len = hop_length * 5

model_name = 'vq.17.vqc'

if platform.node().endswith('.ec2') or platform.node().startswith('ip-'): # Running on EC2
    DATA_PATH = '/home/ubuntu/dataset/lj-16bit'
else:
    DATA_PATH = '/mnt/backup/dataset/lj-16bit'

with open(f'{DATA_PATH}/dataset_ids.pkl', 'rb') as f:
    dataset_ids = pickle.load(f)

#test_ids = dataset_ids[-50:]
#dataset_ids = dataset_ids[:-50]
test_ids = dataset_ids[-3:] + dataset_ids[:3]
dataset_ids = dataset_ids[:-3]

if args.count is not None:
    test_ids = test_ids[:args.count]

dataset = env.AudiobookDataset(dataset_ids, DATA_PATH)

print(f'dataset size: {len(dataset)}')

if args.model is None or args.model == 'vqvae':
    model = vqvae.Model(rnn_dims=896, fc_dims=896,
                  upsample_factors=(4, 4, 4), normalize_vq=True).cuda()
elif args.model == 'wavernn':
    model = wr.Model(rnn_dims=896, fc_dims=896, pad=2,
                  upsample_factors=(5, 5, 11), feat_dims=80).cuda()
else:
    sys.exit(f'Unknown model: {args.model}')

if use_half:
    model = model.half()

paths = env.Paths(model_name, DATA_PATH)

if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
    # Start from scratch
    step = 0
else:
    if args.load:
        prev_model_name = re.sub(r'_[0-9]+$', '', re.sub(r'\.pyt$', '', os.path.basename(args.load)))
        prev_model_basename = prev_model_name.split('_')[0]
        model_basename = model_name.split('_')[0]
        if prev_model_basename != model_basename and not args.force:
            sys.exit(f'refusing to load {args.load} because its basename ({prev_model_basename}) is not {model_basename}')
        paths = env.Paths(prev_model_name, DATA_PATH)
        prev_path = args.load
    else:
        prev_path = paths.model_path()
    step = env.restore(prev_path, model)

optimiser = optim.Adam(model.parameters())

if args.generate:
    model.do_generate(paths, step, DATA_PATH, test_ids, use_half=use_half, verbose=True)#, deterministic=True)
else:
    logger.set_logfile(paths.logfile_path())
    logger.log('------------------------------------------------------------')
    logger.log('-- New training session starts here ------------------------')
    logger.log(time.strftime('%c UTC', time.gmtime()))
    model.do_train(paths, dataset, optimiser, epochs=1000, batch_size=16, seq_len=seq_len, step=step, lr=1e-4, use_half=use_half, valid_ids=test_ids)
