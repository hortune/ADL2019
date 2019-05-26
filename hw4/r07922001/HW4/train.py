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


from utils import load_data
import sys
from collections import namedtuple
from argparse import ArgumentParser
import json

def get_args():
    parser = ArgumentParser()
    parser.add_argument(dest="path", type=str, help="log name")
    args = parser.parse_args()
    return args

args = get_args()
path = args.path
args = json.load(open(os.path.join(path,"config.json")))
args = namedtuple("arg", args.keys())(*args.values())


torch.cuda.set_device(args.dev_num)
######################
# Parameters Setting #
######################

lr_g = args.lr_g
lr_d = args.lr_d
noise_dim = args.noise_dim
feature_dim = args.feature_dim
batch_size = args.batch_size
epoch = args.epoch
seed = args.seed

####################
# Image preprocess #
####################
Images, _, Tags = load_data()
Images = (Images / 127.5) - 1
Images = torch.from_numpy(Images.astype(np.float32)).transpose(1,3)

from loader import T2IDataset

t2IDataset = T2IDataset(Images,Tags)
t2IDLoader = Data.DataLoader(
    dataset = t2IDataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4
)


###################
#   Model   Init  #
###################

from importlib import import_module

if __name__=="__main__":
    if args.mode_name == "WGAN-GP":
        agent = import_module("model_wgan").ACGAN(feature_dim, noise_dim, lr_g, lr_d, args.adv_loss, args)
    elif args.mode_name == "WGAN-RES":
        agent = import_module("model_resgan").ACGAN(feature_dim, noise_dim, lr_g, lr_d, args.adv_loss, args)
    elif args.mode_name == "WGAN-GP-SPEC":
        agent = import_module("model_spectral_wgan").ACGAN(feature_dim, noise_dim, lr_g, lr_d, args.adv_loss, args)
    elif args.mode_name == "WGAN-GP-LAYER":
        agent = import_module("model_layer_wgan").ACGAN(feature_dim, noise_dim, lr_g, lr_d, args.adv_loss, args)
    elif args.mode_name == "WGAN-BERNOULLI":
        agent = import_module("model_wgan_bernoulli").ACGAN(feature_dim, noise_dim, lr_g, lr_d, args.adv_loss, args)
    else:
        assert 1==0
    
    fixNoise = agent.gen_noise(4) 

    gen_iterations = 0

    print ("Start Training")
    try:
        os.makedirs("{}/graph".format(path))
        os.makedirs("{}/model".format(path))
    except:
        pass


    evaluate_embed = torch.Tensor(Tags[:4]).cuda()
    with open(os.path.join(path,"record.txt"),"w") as fd:
        for epoch in range(500):
            i = 0
            while i < len(t2IDLoader):
                data_iter = iter(t2IDLoader)
                d_iters = args.d_iters
                j = 0
                g_iter = args.g_iters
                while j < d_iters and i < len(t2IDLoader) - 1:
                    sample = data_iter.next()
                    right_images = sample['right_images'].cuda()
                    right_embed = sample['right_embed'].cuda()
                    wrong_embed = sample['wrong_embed'].cuda()

                    D_loss = agent.update_D(right_images, right_embed, wrong_embed)
                    j += 1
                    i += 1
                
                j = 0
                while i < len(t2IDLoader) - 1 and j < g_iter:
                    if i != 0:
                        sample = data_iter.next()

                        right_images = sample['right_images'].cuda()
                        right_embed = sample['right_embed'].cuda()
                        wrong_embed = sample['wrong_embed'].cuda()
                        i += 1
                    G_loss = agent.update_G(right_images, right_embed, wrong_embed)
                    gen_iterations += 1
                    print('[%d][%d/%d][%d] Loss_G: %f Loss_D:' % (epoch, i, len(t2IDLoader), gen_iterations, G_loss),D_loss)
                    print('[%d][%d/%d][%d] Loss_G: %f Loss_D:' % (epoch, i, len(t2IDLoader), gen_iterations, G_loss),D_loss,file=fd)
                    j += 1
                    
                if i == len(t2IDLoader) - 1:
                    fake_data = agent.generate(evaluate_embed[:4], fixNoise)
                    fig=plt.figure(figsize=(8, 8))
                    columns = 2
                    rows = 2
                    for gg in range(1, columns*rows +1):
                        fig.add_subplot(rows, columns, gg)
                        plt.imshow(fake_data[gg-1])
                        plt.axis('off')
                    fig.savefig(os.path.join(path,"graph/{}.jpg".format(gen_iterations)))
                    plt.close()
                
                if gen_iterations % 10000 == 0:
                    agent.save(os.path.join(path,'model'), gen_iterations)
                i += 1
