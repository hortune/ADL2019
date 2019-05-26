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
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, cksize):
        super(MeanPoolConv, self).__init__()
        self.netg = nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(input_dim, output_dim, cksize, padding=cksize//2, bias=False)
                )
    def forward(self, x):
        return self.netg(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, size=None):
        super(ResidualBlock, self).__init__()
        if resample == 'down':
            self.csc = MeanPoolConv(input_dim, output_dim, 1)
            conv1    = nn.Conv2d(input_dim, input_dim, 3, padding=3//2, bias=False) # Same
            conv2    = MeanPoolConv(input_dim, output_dim, 3)
            middle_dim = input_dim
        elif resample == 'up':
            self.csc = nn.ConvTranspose2d(input_dim, output_dim, 1, 2, output_padding=1, bias=False)
            conv1    = nn.ConvTranspose2d(input_dim, output_dim, 3, 2, padding=1, output_padding=1, bias=False)
            conv2    = nn.Conv2d(output_dim, output_dim, 3, padding=3//2, bias=False) # Same
            middle_dim = output_dim
        elif resample is None:
            self.csc = nn.Conv2d(input_dim, output_dim, 1, padding=0, bias=False)
            conv1    = nn.Conv2d(input_dim, input_dim, 3, padding=3//2, bias=False)
            conv2    = nn.Conv2d(input_dim, output_dim, 3, padding=3//2, bias=False)
            middle_dim = input_dim
        else:
            raise Exception("GG Your Residual Block's resample method doesn't exist")
        
        self.net = nn.Sequential(
                    nn.BatchNorm2d(input_dim),
                    nn.ReLU(),
                    conv1,
                    nn.BatchNorm2d(middle_dim),
                    nn.ReLU(),
                    conv2,
                )
        self.shortcut = input_dim == output_dim and resample is None
    
    def forward(self, inputs):
        if self.shortcut:
            shortcut = inputs
        else:
            shortcut = self.csc(inputs)
        
        outputs = self.net(inputs)
        
        return outputs + shortcut
        
class Generator(nn.Module):
    def __init__(self,featureDim,noiseDim,hidden_size=64):
        super(Generator, self).__init__()
        self.num_channels = 3
        self.noise_dim = noiseDim
        self.embed_dim = featureDim
        self.latent_dim = self.noise_dim + self.embed_dim
        self.ngf = hidden_size
        
        self.embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.ngf*4*4*32),
            )

        self.netG = nn.Sequential(
            ResidualBlock(self.ngf * 32, self.ngf * 16, 'up'),
            ResidualBlock(self.ngf * 16, self.ngf * 8, 'up'),
            ResidualBlock(self.ngf * 8, self.ngf * 4, 'up'),
            ResidualBlock(self.ngf * 4, self.ngf * 2, 'up'),
            ResidualBlock(self.ngf * 2, self.ngf * 1, 'up'),
            
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf * 1, self.num_channels, 3, padding=3//2, bias=False),
            nn.BatchNorm2d(self.num_channels),
            nn.Sigmoid()
            )
        self.apply(weights_init)

    def forward(self, embed_vector, noise):
        latent_vector = self.embed(torch.cat([embed_vector, noise], 1))
        output = self.netG(latent_vector.view(
            embed_vector.size(0),-1,4,4))
        return output


class Discriminator(nn.Module):
    def __init__(self,FeatureDim, adv_loss, hidden_size=64):
        super(Discriminator,self).__init__()
        
        self.adv_loss = adv_loss
        self.ndf = hidden_size
        
        self.netD = nn.Sequential(
            nn.Conv2d(3, self.ndf, 3, padding=3//2, bias=False),
            ResidualBlock(self.ndf, self.ndf * 2, 'down', 128),
            ResidualBlock(self.ndf * 2, self.ndf * 4, 'down', 64),
            ResidualBlock(self.ndf * 4, self.ndf * 8, 'down', 32),
            ResidualBlock(self.ndf * 8, self.ndf * 16, 'down', 16),
            ResidualBlock(self.ndf * 16, self.ndf * 16, 'down', 8),
            )
        
        self.netD2 = nn.Conv2d(self.ndf*16, 1, 3, 2, 0, bias=False)
        self.netAC = nn.Conv2d(self.ndf*16, 15, 3, 2, 0, bias=False)
        self.apply(weights_init)

    def forward(self, inputs):
        imageC = self.netD(inputs)
        
        if self.adv_loss == "WGAN":
            prob = self.netD2(imageC)
        else:
            prob = torch.sigmoid(self.netD2(imageC))
        cls = torch.sigmoid(self.netAC(imageC))
        return prob.squeeze(), cls.squeeze()

class ACGAN:
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
    
    def clamp(self, model):
        for p in model.parameters():
            p.data.clamp_(-self.args.clamp_rate, self.args.clamp_rate)
    
    def compute_GP(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, 1, 1).cuda()

        interpolates = (alpha * real_data.data + ((1 - alpha) * fake_data.data)).requires_grad_(True)
        disc_interpolates = netD(interpolates)[0]
        gradients = autograd.grad(
                        outputs=disc_interpolates, 
                        inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).cuda().requires_grad_(False),
                        create_graph=True, 
                        retain_graph=True, 
                        only_inputs=True)[0].view(real_data.size(0),-1)
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

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
        # real real
        real_d, cls_real = self.D(right_images)
        
        # train with fake image
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
        elif self.adv_loss == "WGAN":
            D_loss = fake_d.mean() - real_d.mean() +\
                     10*F.binary_cross_entropy(cls_real, right_embed)
            loss = D_loss.item()
            D_loss.backward()
            
            if self.args.restrict == "GP":
                gp = self.compute_GP(self.D, right_images.data, fake.data)
                gp.backward()
            
            self.optimD.step()
            return loss, fake_d.mean() - real_d.mean(), gp.item()
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
        elif self.adv_loss == "WGAN":
            G_loss = -errG.mean()
        else:
            assert 1==0
        G_loss.backward()
        self.optimG.step()
        return G_loss.item()
    
    def generate(self, embedding, noise):
        #fake = ((self.G(embedding, noise) / 2) + 0.5).mul(255)
        fake = self.G(embedding, noise).mul(255)
        fake_data = fake.transpose(1,3) 
        fake_data = fake_data.cpu().data.numpy().astype(np.uint8)
        return fake_data

    def save(self, path, gen_iterations):
        torch.save(self.D.state_dict(), os.path.join(path,'{}D.pt'.format(gen_iterations)))
        torch.save(self.G.state_dict(), os.path.join(path,'{}G.pt'.format(gen_iterations)))
