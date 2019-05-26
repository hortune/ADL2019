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
            embed_vector.size(0),-1,4,4))
        return output


class Discriminator(nn.Module):
    def __init__(self,FeatureDim,hidden_size=32):
        super(Discriminator,self).__init__()
        self.ndf = hidden_size
        self.netD = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False),
                nn.ReLU(True),
                
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
                nn.ReLU(True),
                
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                nn.ReLU(True),
                
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
                nn.ReLU(True),
                
                nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
                nn.ReLU(True),
        )
        #self.projector = Concat_embed(FeatureDim,64) 
        """
        self.netD2 = nn.Sequential(
            nn.Conv2d(hidden_size * 8 + 64, 1, 4, 1, 0, bias=False)
        )
        """

        self.share_w = nn.Linear(self.ndf*16*4*4, 512)
        self.netD2 = nn.Sequential(
            self.share_w,
            nn.ReLU(True),
            nn.Linear(512,1)
        )

        self.netAC = nn.Sequential(
            self.share_w,
            nn.ReLU(True),
            nn.Linear(512,15)
        )
        
        self.apply(weights_init)

    def forward(self, inputs):
        imageC = self.netD(inputs)
        #x = self.projector(imageC,embedding)
        x = imageC.view(inputs.size(0),-1)
        prob = F.sigmoid(self.netD2(x))
        cls = F.sigmoid(self.netAC(x))
        return prob, cls



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

class ACGAN:
    def __init__(self, feature_dim, noise_dim, lr, args):
        self.G = Generator(feature_dim, noise_dim).cuda()
        self.D = Discriminator(feature_dim).cuda()

        self.optimG = optim.Adam(self.G.parameters(), lr=lr,
                                 betas=(0.5,0.999))
        
        self.optimD = optim.Adam(self.D.parameters(), lr=lr,
                                 betas=(0.5,0.999))
        
        self.noise_dim = noise_dim

        one = torch.FloatTensor([1])
        mone = one * -1

        self.one = one.cuda()
        self.mone = mone.cuda()
    
    def update_D(self, right_images, right_embed, *argvs):
        self.D.zero_grad()
        # real real
        real_loss, cls_real = self.D(right_images)
        
        # train with fake image
        noise = torch.bernoulli(
                    torch.empty(right_images.size(0), 
                                self.noise_dim).
                                uniform_(0,1)).cuda()
        
        fake = self.G(right_embed,noise).data
        errD_fake, cls_fake = self.D(fake)
        
        D_loss = F.binary_cross_entropy(errD_fake,
                     torch.zeros(errD_fake.size()).cuda())+\
                     F.binary_cross_entropy(real_loss,
                     torch.ones(real_loss.size()).cuda()) +\
                     F.binary_cross_entropy(cls_real, right_embed)

        D_loss.backward()
        self.optimD.step()
        return D_loss.item()

    def update_G(self, right_images, right_embed, *argvs):
        self.G.zero_grad()
        noise = torch.bernoulli(
                    torch.empty(right_images.size(0), 
                                self.noise_dim).
                                uniform_(0,1)).cuda()
        fake = self.G(right_embed,noise)
        errG, cls_fake = self.D(fake)
        G_loss = F.binary_cross_entropy(errG,torch.ones(errG.size()).cuda()) +\
                 F.binary_cross_entropy(cls_fake, right_embed)
        G_loss.backward()
        #errG = -errG
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
