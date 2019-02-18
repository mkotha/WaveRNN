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
import models.nocond as nc
import models.vqvae as vqvae
import models.wavernn1 as wr
import utils.env as env
import argparse
import platform
import re
import utils.logger as logger
import time
import subprocess

import config

parser = argparse.ArgumentParser(description='Train or run some neural net')
parser.add_argument('--generate', '-g', action='store_true')
parser.add_argument('--float', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load', '-l')
parser.add_argument('--scratch', action='store_true')
parser.add_argument('--model', '-m')
parser.add_argument('--force', action='store_true', help='skip the version check')
parser.add_argument('--count', '-c', type=int, help='number of audio files to generate')
parser.add_argument('--partial', action='append', default=[], help='model to partially load')
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

model_name = 'vq.42.vctk'

with open(f'{config.data_path}/index.pkl', 'rb') as f:
    index = pickle.load(f)

test_index = [x[-1:] if i < 6 else [] for i, x in enumerate(index)]
train_index = [x[:-1] if i < 3 else x for i, x in enumerate(index)]

if args.count is not None:
    test_ids = test_ids[:args.count]

dataset = env.MultispeakerDataset(train_index, config.data_path)

print(f'dataset size: {len(dataset)}')

if args.model is None or args.model == 'vqvae':
    model = vqvae.Model(rnn_dims=896, fc_dims=896, global_decoder_cond_dims=dataset.num_speakers(),
                  upsample_factors=(4, 4, 4), normalize_vq=True, noise_x=True, noise_y=True).cuda()
elif args.model == 'wavernn':
    model = wr.Model(rnn_dims=896, fc_dims=896, pad=2,
                  upsample_factors=(4, 4, 4), feat_dims=80).cuda()
elif args.model == 'nc':
    model = nc.Model(rnn_dims=896, fc_dims=896).cuda()
else:
    sys.exit(f'Unknown model: {args.model}')

if use_half:
    model = model.half()

for partial_path in args.partial:
    model.load_state_dict(torch.load(partial_path), strict=False)

paths = env.Paths(model_name, config.data_path)

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
        if args.generate:
            paths = env.Paths(prev_model_name, config.data_path)
        prev_path = args.load
    else:
        prev_path = paths.model_path()
    step = env.restore(prev_path, model)

#model.freeze_encoder()

optimiser = optim.Adam(model.parameters())

if args.generate:
    model.do_generate(paths, step, config.data_path, test_index, use_half=use_half, verbose=True)#, deterministic=True)
else:
    logger.set_logfile(paths.logfile_path())
    logger.log('------------------------------------------------------------')
    logger.log('-- New training session starts here ------------------------')
    logger.log(time.strftime('%c UTC', time.gmtime()))
    model.do_train(paths, dataset, optimiser, epochs=1000, batch_size=16, seq_len=seq_len, step=step, lr=1e-4, use_half=use_half, valid_index=test_index)
