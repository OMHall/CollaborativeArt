import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from CAN.parameters import *


# custom weights initialization called on netG and netD as in Radford et al (DCGAN) 2015
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# specific uniform cross_entropy for generator
def CrossEntropy_uniform(b_size, output_style, device):
    logsoftmax = nn.LogSoftmax(dim=1)
    unif = torch.full((b_size, n_class), 1/n_class, device = device)
    return torch.mean(-torch.sum(unif * logsoftmax(output_style), 1))

# addition layer of gaussian instance noise 
class GaussianNoise(nn.Module):
    # sigma: sigma*pixel value = stdev of added noise from normal distribution 
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            # scale of noise = stdev of gaussian noise = sigma * pixel value 
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*32)[1024] x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 16, (2,4), 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16)[1024] x 6 x 8        
            # rectangular:
            nn.ConvTranspose2d(ngf * 16, ngf * 8, (2,4), 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8)[512] x 10 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (2,4), 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4)[256] x 18 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2)[128] x 36 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf)[64] x 72 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc)[3] x 144 x 256 (H, W)
            # tanh -> range [-1,1] of the images
        )

    def forward(self, input):
        return self.main(input)


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # apply gaussian noise, sigma 0.1, 
            GaussianNoise(),
            # input is (nc)[3] x 144 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. (ndf)[32] x 72 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. (ndf*2)[64] x 36 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. (ndf*4)[128] x 18 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. (ndf*8)[256] x 9 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. (ndf*16)[512] x 4 x 8
            nn.Conv2d(ndf * 16, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. (ndf*16)[512] x 2 x 4

            # if linear
            nn.Flatten()
        )
        
        self.discriminate = nn.Sequential(
            #nn.Conv2d(ndf * 16, 1, (2,4), 1, 0, bias=False),
            #nn.Sigmoid()

            nn.Linear(ndf * 16 * 4 * 2, 1),
            #nn.Sigmoid()
            )
        
        self.classify = nn.Sequential(
            #nn.Conv2d(ndf * 16, n_class, (2,4), 1, 0, bias=False),

            nn.Linear(ndf * 16 * 4 * 2, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, n_class),
            #nn.Softmax(dim=1)
            )

    def forward(self, input):
        x = self.main(input)
        #print(x.shape)
        d_out = self.discriminate(x)
        c_out = self.classify(x)
        return d_out, c_out
      
