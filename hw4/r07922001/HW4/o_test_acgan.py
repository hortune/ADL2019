import scipy
import matplotlib.pyplot as plt
import numpy as np
import os
plt.switch_backend('agg')

import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random
import torch.utils.data as Data


from utils import load_test_data
from o_model_acgan import Discriminator, Generator
import sys
from PIL import Image

assert len(sys.argv) > 1

######################
# Parameters Setting #
######################

LR = 0.0001
Updates_Proportion = 1
NoiseDim = 200
FeatureDim = 15
BatchSize = 32
Epoch = 300
RandomSeed = 1126

#random.seed(RandomSeed)
#torch.manual_seed(RandomSeed)
#torch.cuda.manual_seed_all(RandomSeed)

##########################
#  Vanilla  Embedding    #
##########################
#embedding = nn.Sequential(nn.Linear(FeatureDim, FeatureDim)).cuda()

##########################
# TAG PreProcess         #
##########################
#Tags = torch.from_numpy(Tags)


####################
# Image preprocess #
####################

Tags = load_test_data(sys.argv[1])
from loader import T2IDataset

t2IDataset = T2IDataset(Tags, Tags)
t2IDLoader = Data.DataLoader(
    dataset = t2IDataset,
    batch_size = BatchSize,
    shuffle = False,
    num_workers = 4
)


###################
#   Model   Init  #
###################

G = Generator(FeatureDim,NoiseDim, 64).cuda()
G.load_state_dict(torch.load('model/200-noise-64-32/model/100000G.pt'))
G = G.eval()

res = []
for data in t2IDLoader:
    right_embed = data['right_embed'].cuda()
    noise = torch.randn(right_embed.size(0),NoiseDim).cuda()
    fake = G(right_embed,noise).mul(255).data
    fake = fake.transpose(1,3).cpu().numpy().astype(np.uint8)
    res.append(fake)
res = np.concatenate(res)

for idx, img in enumerate(res):
    print(idx)
    res = Image.fromarray(img)
    res.save(os.path.join(sys.argv[2],"{}.png".format(idx)))

"""            
fig=plt.figure(figsize=(8, 8))
columns = 5
rows = 2
for gg in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, gg)
    plt.imshow(fake_data[gg-1])
    plt.axis('off')
fig.savefig(os.path.join("model",sys.argv[1],"graph","{}.jpg".format(gen_iterations)))
plt.close()
"""
