# author: Deepankar C.

import argparse
import os
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

import model.networks as networks
from utils.utils import NormalNLLLoss, noise_sample
from data.dataset import MNIST_Dataset, FashionMNIST_Dataset

import pdb

## SETUP
# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # assign random seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# directories
dataroot = "./data"
todaydate = datetime.date.today().strftime('%Y-%m-%d')
saveloc = './results/'+todaydate
saveloc_samples = saveloc+'/samples'
saveloc_models = saveloc+'/models'
saveloc_validation = saveloc+'/validation'
saveloc_logdir = saveloc+'/logs'

if(not os.path.exists(saveloc)):
  # make directories
  os.makedirs(saveloc_samples)
  os.makedirs(saveloc_models)
  os.makedirs(saveloc_validation)
  os.makedirs(saveloc_logdir)

## VARIABLES
# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 62

# latent categorical code
ndc = 10

# continuous categorical code
ncc = 3

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# set GAN training labels
real_label = 1
fake_label = 0

# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
if(ngpu > 0):
  torch.cuda.set_device(0)
  device = torch.device('cuda')

# image transformation
transform = [transforms.ToPILImage(mode='F'), transforms.Resize(image_size), transforms.ToTensor()]

## NETWORKS
# initialize weights
netG = networks.Generator(ndc+ncc+nz, nc, ngf)
netG.apply(networks.weights_init)
print('********* Generator **********\n', netG)

disconvlayers = networks.DisConLayers(nc, 1, ndf)
disconvlayers.apply(networks.weights_init)

netD = networks.Discriminator(ndf)
netD.apply(networks.weights_init)
print('********* Discriminator **********\n', disconvlayers, '\n', netD)

netQ = networks.AuxDistribution(ndf, ndc, ncc)
netQ.apply(networks.weights_init)
print('********* Auxiliary Distribution Q **********\n', disconvlayers, '\n', netQ)

if(ngpu > 0):
  # assign to GPU
  netG = netG.cuda()
  disconvlayers = disconvlayers.cuda()
  netD = netD.cuda()
  netQ = netQ.cuda()

# initialize useful variables
img_list = []
G_losses = {}
D_losses = {}

# initilize losses
criterion = nn.BCELoss()
criterionQ_categorical = nn.CrossEntropyLoss() # discrete latent code
criterionQ_continuous = NormalNLLLoss() # continuous latent code

# initialize optimizers
optimizerG = torch.optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam([{'params': disconvlayers.parameters()}, {'params': netD.parameters()}], lr=lr, betas=(beta1, 0.999))

# load dataset
fashionmnist_dataset = FashionMNIST_Dataset(dataroot, 
                istrain=True, 
                transform=transform)
dataloader = torch.utils.data.DataLoader(fashionmnist_dataset, 
                     batch_size=batch_size, 
                     drop_last=False, 
                     shuffle=True)

# initialize summary writers
epoch_size = (len(fashionmnist_dataset) / batch_size)
writer = SummaryWriter(saveloc_logdir)

# fixed noise for validating generator output
fixed_noise, _ = noise_sample(1, ndc, ncc, nz, batch_size)

if(ngpu > 0):
  fixed_noise = fixed_noise.cuda()

## TRAINING
print("Starting Training Loop...")

for epoch in range(num_epochs):
  # images = next(iter(dataloader))
  for iters, data in enumerate(dataloader):
    # real batch
    real = data
    label = torch.full((batch_size,), real_label)

    if(ngpu > 0):
      real = real.cuda()
      label = label.cuda()

    # update discriminator (real)
    optimizerD.zero_grad()
    con_output = disconvlayers(real)
    output = netD(con_output).view(-1)
    D_losses['real'] = criterion(output, label)
    D_losses['real'].backward()
    D_x = output.mean().item()

    # fake batch
    noise, idx = noise_sample(1, ndc, ncc, nz, batch_size)
    label.fill_(fake_label)

    if(ngpu > 0):
      noise = noise.cuda()

    # pdb.set_trace()
    # update discriminator (fake)
    fake =  netG(noise)
    con_output = disconvlayers(fake.detach())
    output = netD(con_output).view(-1)
    D_losses['fake'] = criterion(output, label)
    D_losses['fake'].backward()
    D_G_z1 = output.mean().item()

    # total discriminator loss
    D_losses['total'] = D_losses['real'] + D_losses['fake']
    optimizerD.step()

    # update generator
    label.fill_(real_label)
    optimizerG.zero_grad()
    con_output = disconvlayers(fake)
    output = netD(con_output).view(-1)
    G_losses['G'] = criterion(output, label)
    D_G_z2 = output.mean().item()

    # ======
    q_logits, q_mu, q_var = netQ(con_output)
    if(ngpu > 0):
      target = torch.cuda.LongTensor(idx)
    else:
      target = torch.LongTensor(idx)

    # loss for discrete latent code
    # pdb.set_trace()
    G_losses['categorical'] = criterionQ_categorical(q_logits, target)

    # loss for continuous latent code.
    G_losses['continuous'] = criterionQ_continuous(noise[:, (nz + ndc):].view(-1, ncc), q_mu, q_var) * 0.1

    # ======
    G_losses['total'] = G_losses['G'] + G_losses['categorical'] + G_losses['continuous']
    G_losses['total'].backward()
    optimizerG.step()

    # output training stats
    if (epoch * epoch_size + iters) % 50 == 0:
      print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' 
        % (epoch, num_epochs, iters, len(dataloader), 
            D_losses['total'].item(), G_losses['total'].item(), D_x, D_G_z1, D_G_z2))

      writer.add_scalars('Probabilities', {'Discriminator-D(x)': D_x, 'Discriminator-D(G(z,c))': D_G_z1, 'Generator-D(G(z,c))': D_G_z2}, epoch * epoch_size + iters)
      writer.add_scalar('DLoss_Fake', D_losses['real'].item(), epoch * epoch_size + iters)
      writer.add_scalar('DLoss_Fake', D_losses['fake'].item(), epoch * epoch_size + iters)
      writer.add_scalar('GLoss_G', G_losses['G'].item(), epoch * epoch_size + iters)
      writer.add_scalar('GLoss_Categorical', G_losses['categorical'].item(), epoch * epoch_size + iters)
      writer.add_scalar('GLoss_Continuous', G_losses['continuous'].item(), epoch * epoch_size + iters)
      writer.add_scalar('GLoss_Total', G_losses['total'].item(), epoch * epoch_size + iters)

    # print images
    if (epoch * epoch_size + iters) % 200 == 0:
      # list of images to show
      choice_idx = np.random.choice(batch_size, 4)

      # create grid of images
      img_grid_real = vutils.make_grid(
        real[choice_idx], 
        nrow=2, 
        normalize=True, 
        range=(0,1)
        )
      img_grid_fake = vutils.make_grid(
        fake[choice_idx], 
        nrow=2, 
        normalize=True, 
        range=(0,1)
        )

      # write to tensorboard
      writer.add_image('MNIST_Real', img_grid_real)
      writer.add_image('MNIST_Fake', img_grid_fake)

    # save images
    if (epoch * epoch_size + iters) % 500 == 0:
      # create grid of images
      vutils.save_image(
        fake, 
        os.path.join(saveloc_samples, str(epoch)+'_'+str(iters)+'.jpg'), 
        nrow=8, 
        normalize=True,
        range=(0,1)
        )

    # Check how the generator is doing by saving G's output on fixed_noise
    if (epoch * epoch_size + iters) % 1000 == 0:
      with torch.no_grad():
        fake = netG(fixed_noise).detach()
        vutils.save_image(
          fake, 
          os.path.join(saveloc_validation, str(epoch)+'_'+str(iters)+'_validation.jpg'), 
          nrow=8,
          normalize=True,
          range=(0.0, 1.0)
          )

    if (epoch * epoch_size + iters) % 5000 == 0:
      modelname_ext = str(epoch)+'_'+str(iters)+'.pt'
      torch.save(netG.state_dict(), saveloc_models+'/netG_'+modelname_ext)
      torch.save(netD.state_dict(), saveloc_models+'/netD_'+modelname_ext)
      torch.save(netQ.state_dict(), saveloc_models+'/netQ_'+modelname_ext)
      torch.save(disconvlayers.state_dict(), saveloc_models+'/disconvlayers_'+modelname_ext)
