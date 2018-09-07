
# coding: utf-8

# ## Alternative Model (Training)
# 
# I've found WaveRNN quite slow to train so here's an alternative that utilises the optimised rnn kernels in Pytorch. The model below is much much faster to train, it will converge in 48hrs when training on 22.5kHz samples (or 24hrs using 16kHz samples) on a single GTX1080. It also works quite well with predicted GTA features. 
# 
# The model is simply two residual GRUs in sequence and then three dense layers with a 512 softmax output. This is supplemented with an upsampling network.
# 
# Since the Pytorch rnn kernels are 'closed', the options for conditioning sites are greatly reduced. Here's the strategy I went with given that restriction:  
# 
# 1 - Upsampling: Nearest neighbour upsampling followed by 2d convolutions with 'horizontal' kernels to interpolate. Split up into two or three layers depending on the stft hop length.
# 
# 2 - A 1d resnet with a 5 wide conv input and 1x1 res blocks. Not sure if this is necessary, but the thinking behind it is: the upsampled features give a local view of the conditioning - why not supplement that with a much wider view of conditioning features, including a peek at the future. One thing to note is that the resnet is computed only once and in parallel, so it shouldn't slow down training/generation much. 
# 
# There's a good chance this model needs regularisation since it overfits a little, so for now train it to ~500k steps for best results.

# In[1]:


import matplotlib.pyplot as plt
import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable 
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
from dsp import *


# In[2]:


bits = 16


# In[3]:


seq_len = hop_length * 5
#seq_len


# In[4]:


# In[6]:


MODEL_PATH = f'model_checkpoints/{notebook_name}.pyt'
DATA_PATH = sys.argv[1]
STEP_PATH = f'model_checkpoints/{notebook_name}_step.npy'
GEN_PATH = f'model_outputs/{notebook_name}/'
os.mkdir(GEN_PATH)


# ## Dataset

# In[7]:


with open(f'{DATA_PATH}dataset_ids.pkl', 'rb') as f:
    dataset_ids = pickle.load(f)

# In[8]:


test_ids = dataset_ids[-50:]
dataset_ids = dataset_ids[:-50]


# In[9]:


class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids
        
    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f'{self.path}mel/{file}.npy')
        x = np.load(f'{self.path}quant/{file}.npy')
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
    wave16 = np.stack(coarse).astype(np.int64)
    coarse = wave16 // 256
    fine = wave16 % 256
    
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    fine = torch.LongTensor(fine)
    
    coarse_f = coarse.float() / 127.5 - 1.
    fine_f = fine.float() / 127.5 - 1.

    x = torch.cat([coarse_f[:, :-1], fine_f[:, :-1], coarse_f[:, 1:]], dim=1)

    return x, mels, coarse[:, 1:], fine[:, 1:]


# In[11]:


dataset = AudiobookDataset(dataset_ids, DATA_PATH)


# In[12]:


#data_loader = DataLoader(dataset, collate_fn=collate, batch_size=32, 
#                         num_workers=0, shuffle=True)


# In[13]:


#len(dataset)


# In[14]:


#x, m, y = next(iter(data_loader))
#x.shape, m.shape, y.shape


# In[15]:


#plot(x.numpy()[0]) 
#plot(y.numpy()[0])
#plot_spec(m.numpy()[0])


# ## Define Model Classes

# In[17]:


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
        self.mask = torch.cat([i2h_mask, i2h_mask, i2h_mask], dim=0)
        num_params(self)
    
    def forward(self, x, mels) :
        mels, aux = self.upsample(mels)
        
        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        
        x = torch.cat([mels, a1, x.unsqueeze(-1)], dim=2)
        h, _ = self.rnn(x)

        h_c, h_f = torch.split(h, self.half_rnn_dims, dim=2)

        o_c = F.relu(self.fc1(torch.cat([h_c, a2], dim=2)))
        p_c = F.log_softmax(self.fc2(o_c), dim=2)

        o_f = F.relu(self.fc3(torch.cat([h_c, a3], dim=2)))
        p_f = F.log_softmax(self.fc4(o_f), dim=2)

        return (p_c, p_f)
    
    def after_update(self):
        self.rnn.gru.weight_ih_l0.data = self.rnn.gru.weight_ih_l0.data * self.mask

    def preview_upsampling(self, mels) :
        mels, aux = self.upsample(mels)
        return mels, aux
    
    def generate(self, mels, save_path) :
        self.eval()
        output = []
        rnn_cell = self.get_gru_cell(self.rnn1)
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

                x = FloatTensor([[c_val, f_val, 0]])

                x = torch.cat([m_t, a1_t, x], dim=1)
                h_0 = rnn1(x, h)

                h_c, _ = torch.split(h_0, self.half_rnn_dims, dim=1)
                
                o_c = F.relu(self.fc1(torch.cat([x, a2_t], dim=1)))
                c_cat = torch.argmax(self.fc2(o_c))
                c_val_new = c_cat[0, 0] / 127.5 - 1.0
                
                x = FloatTensor([[c_val, f_val, c_val_new]])

                x = torch.cat([m_t, a1_t, x], dim=1)
                h = rnn1(x, h)

                _, h_f = torch.split(h, self.half_rnn_dims, dim=1)
                
                o_f = F.relu(self.fc3(torch.cat([x, a3_t], dim=1)))
                f_cat = torch.argmax(self.fc4(o_f))
                f_val = f_cat[0, 0] / 127.5 - 1.0

                c_val = c_val_new
                
                sample = c_cat * 256 + f_cat
                output.append(sample)
                if i % 100 == 0 :
                    speed = int((i + 1) / (time.time() - start))
                    display('%i/%i -- Speed: %i samples/sec', (i + 1, seq_len, speed))
        output = torch.stack(output).cpu().numpy()
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

# ## Instantiate and Check Upsampling

# In[21]:


#hop_length / 11 / 5


# In[22]:


model = Model(rnn_dims=512, fc_dims=512, bits=bits, pad=2,
              upsample_factors=(5, 5, 11), feat_dims=80,
              compute_dims=128, res_out_dims=128, res_blocks=10).cuda()


# In[23]:


if not os.path.exists(MODEL_PATH):
    torch.save(model.state_dict(), MODEL_PATH) 
model.load_state_dict(torch.load(MODEL_PATH)) 


# In[24]:


#mels, aux = model.preview_upsampling(m.cuda())


# In[25]:


#plot_spec(m[0].numpy())
#plot_spec(mels[0].cpu().detach().numpy().T)
#plot_spec(aux[0].cpu().detach().numpy().T)


# In[26]:


global step
step = 0


# In[27]:


if not os.path.exists(STEP_PATH):
    np.save(STEP_PATH, step)
step = np.load(STEP_PATH)
step


# In[28]:


def train(model, optimiser, epochs, batch_size, classes, seq_len, step, lr=1e-4) :
    
    loss_threshold = 4.0
    
    for p in optimiser.param_groups : p['lr'] = lr
    criterion = nn.NLLLoss().cuda()
    
    for e in range(epochs) :

        trn_loader = DataLoader(dataset, collate_fn=collate, batch_size=batch_size, 
                                num_workers=2, shuffle=True, pin_memory=True)
    
        running_loss = 0.
        val_loss = 0.
        start = time.time()
        running_loss = 0.

        iters = len(trn_loader)

        for i, (x, m, y_coarse, y_fine) in enumerate(trn_loader) :
            
            x, m, y_coarse, y_fine = x.cuda(), m.cuda(), y_coarse.cuda(), y_fine.cuda()

            p_c, p_f = model(x, m)
            loss_c = criterion(p_c.transpose(1, 2).unsqueeze(-1), y_coarse.transpose(1, 2).unsqueeze(-1))
            loss_f = criterion(p_f.transpose(1, 2).unsqueeze(-1), y_fine.transpose(1, 2).unsqueeze(-1))
            loss = loss_c + loss_f

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            model.after_update()
            
            speed = (i + 1) / (time.time() - start)
            avg_loss = running_loss / (i + 1)
            
            step += 1
            k = step // 1000
            display('Epoch: %i/%i -- Batch: %i/%i -- Loss: %.3f -- Speed: %.2f steps/sec -- Step: %ik ', 
                   (e + 1, epochs, i + 1, iters, avg_loss, speed, k))
        
        torch.save(model.state_dict(), MODEL_PATH)
        np.save(STEP_PATH, step)
        print(' <saved>')


# In[29]:


optimiser = optim.Adam(model.parameters())


# In[ ]:


train(model, optimiser, epochs=1000, batch_size=16, classes=2**bits, 
      seq_len=seq_len, step=step, lr=1e-4)


# ## Generate Samples

# In[31]:


def generate(samples=3) :
    global output
    k = step // 1000
    test_mels = [np.load(f'{DATA_PATH}mel/{id}.npy') for id in test_ids[:samples]]
    ground_truth = [np.load(f'{DATA_PATH}quant/{id}.npy') for id in test_ids[:samples]]
    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)) :
        print('\nGenerating: %i/%i' % (i+1, samples))
        gt = 2 * gt.astype(np.float32) / (2**bits - 1.) - 1.
        librosa.output.write_wav(f'{GEN_PATH}{k}k_steps_{i}_target.wav', gt, sr=sample_rate)
        output = model.generate(mel, f'{GEN_PATH}{k}k_steps_{i}_generated.wav')


# In[32]:


#generate()


# In[55]:


#plot(output)

