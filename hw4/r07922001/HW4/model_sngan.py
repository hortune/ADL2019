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
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self,featureDim,noiseDim,hidden_size=64):
        super(Generator, self).__init__()
        self.num_channels = 3
        self.noise_dim = noiseDim
        self.embed_dim = featureDim
        self.latent_dim = self.noise_dim + self.embed_dim
        self.ngf = hidden_size
        
        #self.embed = nn.Sequential(
        #    nn.Linear(self.latent_dim,self.latent_dim*4*4),
        #    nn.BatchNorm1d(self.latent_dim*4*4),
        #    nn.ReLU(True)
        #    )
        
        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 32, self.ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
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
            nn.ConvTranspose2d(self.ngf * 1, self.num_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_channels),
            nn.Tanh()
            )
        self.apply(weights_init)

    def forward(self, embed_vector, noise):
        latent_vector = torch.cat([embed_vector, noise], 1)
        output = self.netG(latent_vector.view(
            embed_vector.size(0),self.latent_dim,1,1))
        return output


class Discriminator(nn.Module):
    def __init__(self,FeatureDim, adv_loss, hidden_size=64):
        super(Discriminator,self).__init__()
        
        self.adv_loss = adv_loss
        self.ndf = hidden_size
        self.conv1 = nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(self.ndf*8, self.ndf*16, 4, 2, 1, bias=False)
        
        self.netD = nn.Sequential(
                    self.conv1,
                    nn.LeakyReLU(0.2, True),
                    self.conv2,
                    nn.LeakyReLU(0.2, True),
                    self.conv3,
                    nn.LeakyReLU(0.2, True),
                    self.conv4,
                    nn.LeakyReLU(0.2, True),
                    self.conv5,
                    nn.LeakyReLU(0.2, True),
                )

        self.netD2 = nn.Conv2d(self.ndf*16, 1, 4, 2, 0, bias=False)
        
        self.netAC = nn.Conv2d(self.ndf*16, 15, 4, 2, 0, bias=False)
        """
        self.share_w = nn.Linear(self.ndf*16*4*4, 512)
        self.p1 = nn.Linear(512,1)
        self.pc = nn.Linear(512,15)
        self.netD2 = nn.Sequential(
            self.share_w,
            nn.ReLU(True),
            self.p1
        )

        self.netAC = nn.Sequential(
            self.share_w,
            nn.ReLU(True),
            self.pc
        )
        """
        self.apply(weights_init)

    def forward(self, inputs):
        imageC = self.netD(inputs)
        #x = imageC.view(inputs.size(0),-1)
        if self.adv_loss == "WGAN":
            prob = self.netD2(imageC)
        else:
            prob = F.sigmoid(self.netD2(imageC))
        cls = F.sigmoid(self.netAC(imageC))
        return prob, cls

class ACGAN:
    def __init__(self, feature_dim, noise_dim, lr, adv_loss, args):
        self.G = Generator(feature_dim, noise_dim, args.hidden_dim).cuda()
        self.D = Discriminator(feature_dim, adv_loss, args.hidden_dim).cuda()
        """
        self.optimG = optim.RMSprop(self.G.parameters(), lr=lr)
        self.optimD = optim.RMSprop(self.D.parameters(), lr=lr)
        """
        self.optimG = optim.Adam(self.G.parameters(), lr=lr,
                                 betas=(0.5,0.999))
        
        self.optimD = optim.Adam(self.D.parameters(), lr=lr,
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
        #alpha = alpha.expand(real_data.view(-1, 3 * 128 * 128).size()).cuda()
        #alpha = alpha.expand(real_data.size())
        #alpha = alpha.view(-1, 3, 128, 128)

        interpolates = (alpha * real_data.data + ((1 - alpha) * fake_data.data)).requires_grad_(True)
        disc_interpolates = netD(interpolates)[0]
        gradients = autograd.grad(
                        outputs=disc_interpolates, 
                        inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).cuda().requires_grad_(False),
                        create_graph=True, 
                        retain_graph=True, 
                        only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.args.gp_coef
        return gradient_penalty

    def gen_noise(self, size, loss="bernoulli"):
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
            exit()
            D_loss = F.binary_cross_entropy(fake_d,
                         torch.zeros(fake_d.size()).cuda())+\
                         F.binary_cross_entropy(real_d,
                         torch.ones(real_d.size()).cuda()) +\
                         F.binary_cross_entropy(cls_real, right_embed)
        elif self.adv_loss == "WGAN":
            D_loss = fake_d.mean() - real_d.mean() +\
                     F.binary_cross_entropy(cls_real, right_embed)
        else:
            assert 1 == 0
        
        loss = D_loss.item()
        
        if self.args.restrict == "GP":
            gp = self.compute_GP(self.D, right_images.data, fake.data)
            D_loss += gp
            loss += gp.item()
        
        D_loss.backward()
        self.optimD.step()
        #self.clamp(self.D)
        return loss

    def update_G(self, right_images, right_embed, *argvs):
        self.G.zero_grad()
        
        noise = self.gen_noise(right_images.size(0))
        fake = self.G(right_embed,noise)
        errG, cls_fake = self.D(fake)

        if self.adv_loss == "DCGAN":
            exit()
            G_loss = F.binary_cross_entropy(errG,torch.ones(errG.size()).cuda()) +\
                     F.binary_cross_entropy(cls_fake, right_embed)
        elif self.adv_loss == "WGAN":
            G_loss = -errG.mean() +\
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
