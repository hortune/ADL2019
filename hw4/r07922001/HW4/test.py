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
import sys
from collections import namedtuple
from argparse import ArgumentParser
import json

from PIL import Image
import torchvision.utils as vutils
def get_args():
    parser = ArgumentParser()
    parser.add_argument("-o",dest="out_dir", type=str, required=True, help="output dir")
    parser.add_argument("-i",dest="lab_loc", type=str, required=True, help="label location")
    args = parser.parse_args()
    return args

args = get_args()
path = "model/WGAN-FINAL" #args.path

lab_loc = args.lab_loc
out_dir = args.out_dir

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
batch_size = args.test_batch_size
epoch = args.epoch
seed = args.seed

####################
# Image preprocess #
####################
Tags = load_test_data(lab_loc)

from loader import T2IDataset

t2IDataset = T2IDataset(Tags,Tags)
t2IDLoader = Data.DataLoader(
    dataset = t2IDataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4
)


###################
#   Model   Init  #
###################

from importlib import import_module

if __name__=="__main__":
    agent = import_module("model_wgan").ACGAN(feature_dim, noise_dim, lr_g, lr_d, args.adv_loss, args)
    """
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
    """
    agent.load_G(os.path.join("checkpoint", "190000G.pt"))
    res = []
    for data in t2IDLoader:
        right_embed = data['right_embed'].cuda()
        noise = agent.gen_noise(right_embed.size(0))
        fake = agent.generate(right_embed, noise, True)
        res.append(fake.data.cpu().transpose(2,3)/255)
    
    res = torch.cat(res)

    for idx, img in enumerate(res):
        vutils.save_image(img, os.path.join(out_dir,"{}.png".format(idx)))

