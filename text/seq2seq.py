#!/usr/bin/env python3
# coding: utf-8
#################################################
# encoder/decoder implementation with attention #
#################################################

# the goal is to implement an influentionnal seq2seq architecture with attention
# we model P(Y|X) of Y=(y_1, ..., y_n) a target sequence, given source sequence X=(x_1, ..., x_n) as in the nmt paper by badhanau

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from IPython.core.debugger import set_trace

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

seq = pad_sequence([torch.tensor([1,2]),
                    torch.tensor([1]),
                    torch.tensor([1,2,3,4])], batch_first=True)
print("seq\n=====")
print(seq)
print(seq.size())
lens = [2,1,4]

vocab_size=10
embedding_dim=64
emb = nn.Embedding(vocab_size, embedding_dim)
print(emb)
print(emb.weight.size())
print("emb\n=====")

embed = emb(seq)
print(embed)
print(embed.size())
print("embed\n=====")


packed = pack_padded_sequence(embed, lens, batch_first=True, enforce_sorted=False)
print(packed)
print(packed.data.size())
print("packed\n=====")


hidden_dim=128
rnn = nn.GRU(embedding_dim, hidden_dim, bidirectional=True)
encoded_hidden, encoded_final = rnn(packed)
print(encoded_hidden)
print(encoded_hidden.data.size())
print("encoded_hidden\n=====")
print(encoded_final)
print(encoded_final.data.size())
print("encoded_final\n=====")

output, _ = pad_packed_sequence(encoded_hidden, batch_first=True)
print(output)
print(output.data.size())
print("output\n=====")

