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
from model_dcgan import Discriminator, Generator
import sys

assert len(sys.argv) > 1

######################
# Parameters Setting #
######################

LR = 0.0002
Updates_Proportion = 1
NoiseDim = 100
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
Images, _, Tags = load_data()
Images = Images / 255 
Images = torch.from_numpy(Images.astype(np.float32)).transpose(1,3)


from loader import T2IDataset

t2IDataset = T2IDataset(Images,Tags)
t2IDLoader = Data.DataLoader(
    dataset = t2IDataset,
    batch_size = BatchSize,
    shuffle = True,
    num_workers = 4
)


###################
#   Model   Init  #
###################

G = Generator(FeatureDim,NoiseDim).cuda()
D = Discriminator(FeatureDim).cuda()

optimG = optim.Adam(G.parameters(), lr =LR,betas=(0.5,0.999))
optimD = optim.Adam(D.parameters(), lr =LR,betas=(0.5,0.999))

one = torch.FloatTensor([1])
mone = one * -1

one = one.cuda()
mone = mone.cuda()

#fixNoise = torch.randn(10,NoiseDim).cuda().detach()
fixNoise = torch.bernoulli(torch.empty(10,NoiseDim).uniform_(0,1)).cuda().detach()

gen_iterations = 0

def compute_GP(netD, real_data, real_embed, fake_data, LAMBDA):
	BATCH_SIZE = real_data.size(0)
	alpha = torch.rand(BATCH_SIZE, 1)
	alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 64, 64)
	alpha = alpha.cuda()

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	interpolates = interpolates.cuda()
	disc_interpolates = netD(interpolates, real_embed)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
		grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
		create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

	return gradient_penalty

print ("Start Training")
if not os.path.exists("model/{}".format(sys.argv[1])):
    os.makedirs("model/{}".format(sys.argv[1]))
    os.makedirs("model/{}/graph".format(sys.argv[1]))
    os.makedirs("model/{}/model".format(sys.argv[1]))


evaluate_embed = None
with open(os.path.join("model",sys.argv[1],"record.txt"),"w") as fd:
    for epoch in range(500):
        i = 0
        while i < len(t2IDLoader):
            ############################
            # (1) Update D network
            ###########################
            data_iter = iter(t2IDLoader)
            for p in D.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
                
            Diters = 1
            j = 0
            while j < Diters:
                j += 1
                # clamp parameters to a cube
                # train with real
                i += 1
                sample = data_iter.next()

                right_images = sample['right_images'].cuda()
                right_embed = sample['right_embed'].cuda()
                wrong_embed = sample['wrong_embed'].cuda()

                if evaluate_embed is None:
                    evaluate_embed = right_embed

                D.zero_grad()
                # real real
                real_loss = D(right_images, right_embed)
                #real_loss.backward(mone)
                
                # fake real
                wrong_loss = D(right_images,wrong_embed)
                #wrong_loss.backward(one)
                
                # train with fake image
                #noise = torch.randn(right_images.size(0),NoiseDim).cuda()
                noise = torch.bernoulli(torch.empty(right_images.size(0),NoiseDim).uniform_(0,1)).cuda()
                fake = G(right_embed,noise).data
                errD_fake = D(fake,right_embed)
                #errD_fake.backward(one)
                
                Total_loss = (F.binary_cross_entropy(errD_fake,
                             torch.zeros(errD_fake.size()).cuda()) +\
                             F.binary_cross_entropy(wrong_loss,
                             torch.zeros(wrong_loss.size()).cuda()))/2+\
                             F.binary_cross_entropy(real_loss,
                             torch.ones(real_loss.size()).cuda())
                Total_loss.backward()
                #gp = compute_GP(D, right_images.data, right_embed, fake.data, LAMBDA=10)
                #gp.backward()
                
                #d_loss = real_loss - errD_fake - wrong_loss
                optimD.step()
            ############################
            # (2) Update G network     # 
            ############################
            
            for index in range(2):
                for p in D.parameters():
                    p.requires_grad = False # to avoid computation
                
                if index ==1:
                    i += 1
                    sample = data_iter.next()
                    right_images = sample['right_images'].cuda()
                    right_embed = sample['right_embed'].cuda()
                    wrong_embed = sample['wrong_embed'].cuda()
                
                G.zero_grad()
                # in case our last batch was the tail batch of the dataloader,
                # make sure we feed a full batch of noise
                #noise = torch.randn(right_images.size(0),NoiseDim).cuda()
                noise = torch.bernoulli(torch.empty(right_images.size(0),NoiseDim).uniform_(0,1)).cuda()
                fake = G(right_embed,noise)
                errG = D(fake,right_embed)
                G_loss = F.binary_cross_entropy(errG,torch.ones(errG.size()).cuda())
                G_loss.backward()
                #errG = -errG
                optimG.step()
                gen_iterations += 1
                print('[%d][%d/%d][%d] Loss_G: %f Loss_D: %f' % (epoch, i, len(t2IDLoader), gen_iterations, G_loss.item() ,Total_loss.item()))
                print('[%d][%d/%d][%d] Loss_G: %f Loss_D: %f' % (epoch, i, len(t2IDLoader), gen_iterations, G_loss.item() ,Total_loss.item()),file=fd)
            
            if gen_iterations % 500 == 0:
                
                fake = G(evaluate_embed[:10],fixNoise).mul(255)
                fake_data = fake.transpose(1,3) 
                fake_data = fake_data.cpu().data.numpy().astype(np.uint8)
                
                fig=plt.figure(figsize=(8, 8))
                columns = 5
                rows = 2
                for gg in range(1, columns*rows +1):
                    fig.add_subplot(rows, columns, gg)
                    plt.imshow(fake_data[gg-1])
                    plt.axis('off')
                fig.savefig(os.path.join("model",sys.argv[1],"graph","{}.jpg".format(gen_iterations)))
                plt.close()
            
            if gen_iterations % 10000 == 0:
                torch.save(D.state_dict(),os.path.join("model",sys.argv[1],"model",'{}D.pt'.format(gen_iterations)))
                torch.save(G.state_dict(),os.path.join("model",sys.argv[1],"model",'{}G.pt'.format(gen_iterations)))


torch.save(D.state_dict(),os.path.join("model",sys.argv[1],"finalD.pt"))
torch.save(G.state_dict(),os.path.join("model",sys.argv[1],'finalG.pt'))
