# author: Deepankar C.

import argparse
import os
import random
import argparse
import pdb

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import numpy as np

from model.networks import Generator
from utils.utils import noise_sample, controlled_noise_sample


def main(args):
  # Set random seed for reproducibility
  seed = args.seed
  if(seed is None):
    seed = random.randint(1, 10000) # use if you want new results
  print("Random Seed: ", seed)
  random.seed(seed)
  torch.manual_seed(seed)

  # directories
  saveloc = os.path.join(args.saveloc, args.expname)
  modelpath = os.path.join(args.modelpath, args.modelname)
  if(not os.path.exists(saveloc)):
    os.makedirs(saveloc)

  num_batches = 1 # no. of image batches to generate
  batch_size = 200 # no. of images to generate
  nc = 1 # Number of channels in the training images. For color images this is 3
  nz = 62 # Size of z latent vector (i.e. size of generator input)
  ndc = 10 # latent categorical code
  ncc = 3 # continuous categorical code
  ngf = 64
  fixed_exp = False

  # Number of GPUs available. Use 0 for CPU mode.
  ngpu = 1
  if(ngpu > 0):
    torch.cuda.set_device(0)

  # load model weights
  netG = Generator(ndc+ncc+nz, nc, ngf)
  print('********* Generator **********\n', netG)
  netG.load_state_dict(torch.load(modelpath))
  netG.eval()

  if(ngpu > 0):
    # assign to GPU
    netG = netG.cuda()

  print("Starting Testing Loop...")

  if(fixed_exp):
    z_rand = torch.randn((batch_size, nz, 1, 1))
    z_disc = torch.LongTensor(np.random.randint(ndc, size=(batch_size, 1)))
    z_cont = torch.rand((batch_size, ncc, 1, 1)) * 2 - 1

    # multiple digits plot
    # z_cont2 = torch.tensor(np.tile(np.linspace(-1, 1, 20).reshape(1, -1), reps=(10, 1))).view(batch_size, -1, 1, 1)
    # z_cont1 = torch.rand((batch_size, 1, 1, 1)) * 2 - 1
    # pdb.set_trace()

    # z_disc = torch.LongTensor(np.tile(np.arange(0, 10).reshape(-1,1), reps=[1, batch_size // 10]))
    # z_disc = torch.LongTensor(np.repeat(np.arange(0, 10), repeats=batch_size // 10)).reshape(-1,1)
    # z_cont = torch.tensor(np.tile(np.linspace(-1, 1, 7).reshape(1,-1), reps=[10, 1]))
    # z_disc = 3 * torch.ones((batch_size, 1), dtype=torch.long)
    # z_cont2 = torch.linspace(-6, 6, batch_size).view(batch_size, -1, 1, 1)
    # z_cont1 = torch.rand((batch_size, 1, 1, 1)) * 2 - 1
    
    # z_cont12 = torch.rand((batch_size, 2, 1, 1)) * 2 - 1
    # z_cont3 = torch.linspace(-5, 5, batch_size).view(batch_size, -1, 1, 1)
    # z_cont4 = torch.rand((batch_size, 1, 1, 1)) * 2 - 1

    # z_cont2 = torch.linspace(-4, 4, batch_size).view(batch_size, -1, 1, 1)
    # z_cont1 = torch.rand((batch_size, 1, 1, 1)) * 2 - 1
    # z_cont2 =  torch.tensor(np.tile(np.linspace(-2.5, 2.5, 20).reshape(1, -1), reps=(10, 1))).view(batch_size, -1, 1, 1)
    # z_cont3 = torch.rand((batch_size, 1, 1, 1)) * 2 - 1
    # z_cont3 = torch.rand((batch_size, 1, 1, 1)) * 2 - 1 # torch.linspace(-2, 2, batch_size).view(batch_size, -1, 1, 1)

    # z_cont = torch.cat([
    #   z_cont1.type(torch.float32), 
    #   z_cont2.type(torch.float32), 
    #   z_cont3.type(torch.float32)], 
    #   z_cont4.type(torch.float32)], 
    #   dim=1
    #   )

  for iters in range(num_batches):
    # fake batch
    if(fixed_exp):
      noise, idx = controlled_noise_sample(
        batch_size, ndc, 
        z_random = z_rand,
        # nz=nz, 
        z_categorical=z_disc, 
        # num_discrete = 1,
        z_continuous=z_cont
        # num_continuous=ncc, 
        )
    else:
      noise, idx = noise_sample(1, ndc, ncc, nz, 1)
    if(ngpu > 0):
      noise = noise.cuda()

    fake =  netG(noise)

    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
      fake = netG(noise).detach()
      vutils.save_image(
        fake, 
        os.path.join(saveloc, str(iters)+'.jpg'), 
        nrow=20,
        normalize=True,
        range=(0.0, 1.0)
        )

    with open(os.path.join(saveloc, 'metadata.txt'), 'a') as f:
      for lineno in range(batch_size):
        if(batch_size == 1):
          f.write('C1: {:1.0f}, '.format(idx.item()))
        else:
          f.write('C1: {:1.0f}, '.format(idx[lineno].item()))
        for i, item in enumerate(noise[lineno, nz+ndc:].squeeze()):
          f.write('C'+str(2+i)+': {:1.4f}, '.format(item.item()))
        f.write('\n')

    print('Generated file {}'.format(iters))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Perform inference using InfoGAN')
  parser.add_argument('--modelpath', type=str, default='./results/2019-12-13 (fashionmnist)/models', help='path to the saved models')
  parser.add_argument('--modelname', type=str, help='name of saved models', required=True)
  parser.add_argument('--saveloc', type=str, default='./results/2019-12-13 (fashionmnist)', help='save location of images')
  parser.add_argument('--expname', type=str, default='voNoise', help='name of experiment', required=True)
  parser.add_argument('--seed', type=int, default=42, help='RNG seed')

  args = parser.parse_args()
  main(args)