import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import torch.autograd as autograd
import os
import torch.nn.utils.spectral_norm as spectral_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'MeanPoolConv':
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, cksize):
        super(MeanPoolConv, self).__init__()
        self.netg = nn.Sequential(
                    nn.AvgPool2d(2),
                    spectral_norm(nn.Conv2d(input_dim, output_dim, cksize, padding=cksize//2, bias=False))
                )
    def forward(self, x):
        return self.netg(x)
        
class Generator(nn.Module):
    def __init__(self,featureDim,noiseDim,hidden_size=32):
        super(Generator, self).__init__()
        self.num_channels = 3
        self.noise_dim = noiseDim
        self.embed_dim = featureDim
        self.latent_dim = self.noise_dim +  self.embed_dim
        self.ngf = hidden_size
        
        self.embed = nn.Sequential(
            nn.Linear(self.latent_dim,self.latent_dim*4*4),
            nn.BatchNorm1d(self.latent_dim*4*4),
            nn.ReLU(True)
            )
        
        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 1,self.num_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_channels),
            nn.Sigmoid()
            )
        self.apply(weights_init)
    
    def forward(self, embed_vector,noise):
        latent_vector = self.embed(torch.cat([embed_vector, noise], 1))
        output = self.netG(latent_vector.view(
            embed_vector.size(0),self.latent_dim,4,4))
        return output

class Discriminator(nn.Module):
    def __init__(self,FeatureDim, adv_loss, hidden_size=64):
        super(Discriminator,self).__init__()
        self.adv_loss = adv_loss
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
        if self.adv_loss == "WGAN":
            prob = self.netD2(imageC)
        else:
            prob = torch.sigmoid(self.netD2(imageC))
        cls = torch.sigmoid(self.netAC(imageC))
        return prob.squeeze(), cls.squeeze()

class NWGAN:
    def __init__(self, feature_dim, noise_dim, lr_g, lr_d, adv_loss, args):
        self.G = Generator(feature_dim, noise_dim, args.hidden_dim).cuda()
        self.D = Discriminator(feature_dim, adv_loss, args.hidden_dim).cuda()
        
        """
        self.optimG = optim.RMSprop(self.G.parameters(), lr=lr)
        
        self.optimD = optim.RMSprop(self.D.parameters(), lr=lr)
        
        """
        self.optimG = optim.Adam(self.G.parameters(), lr=lr_g,
                                 betas=(0.5,0.999))
        
        self.optimD = optim.Adam(self.D.parameters(), lr=lr_d,
                                 betas=(0.5,0.999))
        self.noise_dim = noise_dim
        self.adv_loss = adv_loss
        one = torch.FloatTensor([1])
        mone = one * -1

        self.one = one.cuda()
        self.mone = mone.cuda()
        self.args = args
    
    def gen_noise(self, size, loss="normal"):
        if loss == "normal":
            return torch.empty(size, self.noise_dim).normal_(0,1).cuda()
        else:
            return torch.bernoulli(
                    torch.empty(size, 
                                self.noise_dim).
                                uniform_(0,1)).cuda()

    def update_D(self, right_images, right_embed, *argvs):
        self.D.zero_grad()
        real_d, cls_real = self.D(right_images)
        noise = self.gen_noise(right_images.size(0))
        fake = self.G(right_embed,noise)
        fake_d, cls_fake = self.D(fake)
        
        if self.adv_loss == "DCGAN":
            D_loss = F.binary_cross_entropy(fake_d,
                         torch.zeros(fake_d.size()).cuda())+\
                         F.binary_cross_entropy(real_d,
                         torch.ones(real_d.size()).cuda()) +\
                         F.binary_cross_entropy(cls_real, right_embed)
            D_loss.backward()
            self.optimD.step()
            return D_loss.item()
        else:
            assert 1 == 0

    def update_G(self, right_images, right_embed, *argvs):
        self.G.zero_grad()
        
        noise = self.gen_noise(right_images.size(0))
        fake = self.G(right_embed,noise)
        errG, cls_fake = self.D(fake)

        if self.adv_loss == "DCGAN":
            G_loss = F.binary_cross_entropy(errG,torch.ones(errG.size()).cuda()) +\
                     F.binary_cross_entropy(cls_fake, right_embed)
        else:
            assert 1==0
        G_loss.backward()
        self.optimG.step()
        return G_loss.item()
    
    def generate(self, embedding, noise):
        fake = self.G(embedding, noise).mul(255)
        fake_data = fake.transpose(1,3) 
        fake_data = fake_data.cpu().data.numpy().astype(np.uint8)
        return fake_data

    def save(self, path, gen_iterations):
        torch.save(self.D.state_dict(), os.path.join(path,'{}D.pt'.format(gen_iterations)))
        torch.save(self.G.state_dict(), os.path.join(path,'{}G.pt'.format(gen_iterations)))
