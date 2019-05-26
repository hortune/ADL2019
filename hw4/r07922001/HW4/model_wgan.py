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

class Generator(nn.Module):
    def __init__(self,featureDim,noiseDim,hidden_size=64):
        super(Generator, self).__init__()
        self.num_channels = 3
        self.noise_dim = noiseDim
        self.ngf = hidden_size
        
        self.noise_embed = nn.Sequential(
            nn.Linear(self.noise_dim, self.ngf*4*4),
        )

        self.label_embed = nn.Linear(15, self.ngf*8)

        self.noise_conv = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            )

        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 1),
            nn.ReLU(True),
           
            nn.ConvTranspose2d(self.ngf * 1, self.num_channels, 4, 2, 1),
            nn.Tanh()
            )
        self.apply(weights_init)

    def forward(self, embed_vector, noise):
        noise = self.noise_embed(noise).view(noise.size(0), -1, 4, 4)
        embed_vector = self.label_embed(embed_vector)
        embed_vector = embed_vector.unsqueeze(2).expand(
                                                    noise.size(0),
                                                    self.ngf*8,
                                                    64).view(-1,self.ngf*8,8,8)
        noise = self.noise_conv(noise)
        latent_vector = torch.cat([embed_vector, noise], dim=1)
        output = self.netG(latent_vector)
        return output

class Discriminator(nn.Module):
    def __init__(self,FeatureDim, adv_loss, hidden_size=64):
        super(Discriminator,self).__init__()
        
        self.adv_loss = adv_loss
        self.ndf = hidden_size
        self.label_embed = nn.Linear(15, self.ndf*8)
        
        self.conv1 = nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False)
        self.conv5 = nn.Sequential(
                        nn.Conv2d(self.ndf*8*2, self.ndf*1, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, True)
                        )
        
        self.netD = nn.Sequential(
                    self.conv1,
                    nn.BatchNorm2d(self.ndf),
                    nn.LeakyReLU(0.2, True),
                    self.conv2,
                    nn.BatchNorm2d(self.ndf*2),
                    nn.LeakyReLU(0.2, True),
                    self.conv3,
                    nn.BatchNorm2d(self.ndf*4),
                    nn.LeakyReLU(0.2, True),
                    self.conv4,
                    nn.BatchNorm2d(self.ndf*8),
                    nn.LeakyReLU(0.2, True),
                )
        
        self.proj = nn.Linear(self.ndf*4*4,  self.ndf, bias=False)
        self.netD2 = nn.Linear(self.ndf,1, bias=False)
        self.netAC = nn.Linear(self.ndf,15, bias=False)
        self.apply(weights_init)

    def forward(self, inputs, label):
        label = self.label_embed(label)
        label = label.unsqueeze(2).expand(
                                        label.size(0),-1,64).view(
                                            label.size(0),-1,8,8)
        imageC = self.netD(inputs)
        x = torch.cat([imageC, label], dim=1)
        x = self.conv5(x).view(label.size(0), -1)
        x = self.proj(x)
        prob = self.netD2(x)
        cls = self.netAC(x)
        return prob.squeeze(), cls.squeeze()
class ACGAN:
    def __init__(self, feature_dim, noise_dim, lr_G, lr_D, adv_loss, args):
        self.G = Generator(feature_dim, noise_dim, args.hidden_dim).cuda()
        self.D = Discriminator(feature_dim, adv_loss, args.hidden_dim).cuda()
        self.optimG = optim.Adam(self.G.parameters(), lr=lr_G,
                                 betas=(0.5,0.999))
        
        self.optimD = optim.Adam(self.D.parameters(), lr=lr_D,
                                 betas=(0.5,0.999))
        self.noise_dim = noise_dim
        self.adv_loss = adv_loss
        one = torch.FloatTensor([1])
        mone = one * -1

        self.one = one.cuda()
        self.mone = mone.cuda()
        self.args = args
        self.G.train()
        self.D.train()
        for p in self.D.parameters():
            p.data.clamp_(-0.01,0.01)

    def load_G(self, path):
        self.G.load_state_dict(torch.load(path, map_location={'cuda:1':'cuda:0'}))
        self.G = self.G.eval()
    
    def compute_GP(self, netD, real_data, fake_data, right_embed):
        alpha = torch.rand(real_data.size(0), 1, 1, 1).cuda()
        interpolates = autograd.Variable(((alpha * real_data + ((1 - alpha) * fake_data))).contiguous(), requires_grad=True)
        disc_interpolates = netD(interpolates, right_embed)[0]
        gradients = autograd.grad(
                        outputs=disc_interpolates, 
                        inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                        create_graph=True, 
                        retain_graph=True, 
                        only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.args.gp_coef
        return gradient_penalty
    
    def gen_noise(self, size, loss="normal"):
        if loss == "normal":
            return torch.empty(size, self.noise_dim).normal_(0,1).cuda()
        else:
            return torch.bernoulli(
                    torch.empty(size, 
                                self.noise_dim).
                                uniform_(0,1)).cuda()

    def update_D(self, right_images, right_embed, fake_embed, *argvs):
        self.D.zero_grad()
        real_d, cls_real = self.D(right_images, right_embed)
        noise = self.gen_noise(right_images.size(0))
        fake = self.G(right_embed, noise).data
        fake_d, cls_fake = self.D(fake.detach(), right_embed)
        
        if self.args.fake_loss:
            noise1 = self.gen_noise(right_images.size(0))
            fake1 = self.G(fake_embed, noise1).data
            fake_d1, cls_fake1 = self.D(fake1.detach(), fake_embed)
            
            D_loss = (fake_d1.mean() + fake_d.mean())/2 - real_d.mean() +\
                    F.binary_cross_entropy_with_logits(cls_real, right_embed)*10 +\
                    F.binary_cross_entropy_with_logits(cls_fake, right_embed)*5 +\
                    F.binary_cross_entropy_with_logits(cls_fake1, fake_embed)*5
        else:
            D_loss = fake_d.mean() - real_d.mean() +\
                    F.binary_cross_entropy_with_logits(cls_real, right_embed)*10 +\
                    F.binary_cross_entropy_with_logits(cls_fake, right_embed)*10
        gp = self.compute_GP(self.D, right_images.data, fake.data, right_embed.data)
        self.D.zero_grad()
        loss = D_loss + gp
        loss.backward()
        
        self.optimD.step()

        return loss.item(), D_loss.item(), gp.item()

    def update_G(self, right_images, right_embed, fake_embed, *argvs):
        self.G.zero_grad()
        
        noise = self.gen_noise(right_images.size(0))
        fake = self.G(right_embed,noise)
        errG, cls_fake = self.D(fake, right_embed)
        
        if self.args.fake_loss:
            noise1 = self.gen_noise(right_images.size(0))
            fake1 = self.G(fake_embed, noise1)
            errG1, cls_fake1 = self.D(fake1, fake_embed)
        
            G_loss = (-errG.mean() - errG1.mean())/2 +\
                     F.binary_cross_entropy_with_logits(cls_fake, right_embed)*5 +\
                     F.binary_cross_entropy_with_logits(cls_fake1, fake_embed)*5
        else:
            G_loss = -errG.mean() +\
                     F.binary_cross_entropy_with_logits(cls_fake, right_embed)*10

        G_loss.backward()
        self.optimG.step()
        return G_loss.item()
    
    def generate(self, embedding, noise, torchvision=False):
        fake_data = self.G(embedding, noise).mul(127.5) + 127.5
        if not torchvision:
            fake_data = fake_data.transpose(1,3) 
            fake_data = fake_data.cpu().data.numpy().astype(np.uint8)
        return fake_data

    def save(self, path, gen_iterations):
        torch.save(self.D.state_dict(), os.path.join(path,'{}D.pt'.format(gen_iterations)))
        torch.save(self.G.state_dict(), os.path.join(path,'{}G.pt'.format(gen_iterations)))
