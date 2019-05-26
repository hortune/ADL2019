import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self,featureDim,noiseDim,hidden_size=32):
        super(Generator, self).__init__()
        self.num_channels = 3
        self.noise_dim = noiseDim
        self.embed_dim = featureDim
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim +  self.embed_dim #self.projected_embed_dim
        self.ngf = hidden_size
        
        self.embed = nn.Sequential(
            nn.Linear(self.latent_dim,self.latent_dim*4*4),
            nn.BatchNorm1d(self.latent_dim*4*4),
            nn.ReLU(True)
            )
        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 1,self.num_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_channels),
            nn.Sigmoid()
             # state size. (num_channels) x 64 x 64
            )
        self.apply(weights_init)

    def forward(self, embed_vector,noise):
        latent_vector = self.embed(torch.cat([embed_vector, noise], 1))
        output = self.netG(latent_vector.view(
            embed_vector.size(0),self.latent_dim,4,4))
        return output

class Discriminator(nn.Module):
    def __init__(self,FeatureDim, hidden_size=64):
        super(Discriminator,self).__init__()
        self.ndf = hidden_size
        self.netD = nn.Sequential(
                nn.Conv2d(3, self.ndf, 3, padding=1, bias=False),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(self.ndf, self.ndf * 2, 3, padding=1, bias=False),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, padding=1, bias=False),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 3, padding=1, bias=False),
                nn.ReLU(True),
                nn.AvgPool2d(2),
                nn.Conv2d(self.ndf * 8, self.ndf * 16, 3, padding=1, bias=False),
                nn.ReLU(True),
                nn.AvgPool2d(2),
        )

        self.sw =  nn.Sequential(
            nn.Linear(self.ndf*16*4*4, 512),
            nn.ReLU(True)
            )
        
        self.netD2 = nn.Linear(512, 1)
        self.netAC    = nn.Linear(512, 15)
        self.apply(weights_init)
    
    def forward(self, inputs):
        imageC = self.netD(inputs).view(inputs.size(0), -1)
        imageC = self.sw(imageC)
        prob = torch.sigmoid(self.netD2(imageC))
        cls = torch.sigmoid(self.netAC(imageC))
        return prob.squeeze(), cls.squeeze()


class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, projected_embed_dim),
            nn.BatchNorm1d(projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)
        return hidden_concat
