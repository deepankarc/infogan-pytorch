# author: Deepankar C. 

import torch
import numpy as np

# def noise_sample(num_discrete, dim_discrete, num_continuous, nz, batch_size):
#   # noise vector
#   z = torch.randn((batch_size, nz, 1, 1))

#   # random categorical variable
#   if(num_discrete != 0):
#     idx = torch.tensor(np.random.randint(dim_discrete, size=(batch_size, 1)))
#     z_categorical = torch.zeros((batch_size, dim_discrete))
#     z_categorical.scatter_(1, idx, 1.0)
#     z_categorical = z_categorical.view(batch_size, -1, 1, 1)
#   # random continuous variable (uniform between -1 and 1)
#   if(num_continuous != 0):
#     z_continuous = torch.rand((batch_size, num_continuous, 1, 1)) * 2 - 1

#   noise = torch.cat((z, z_categorical, z_continuous), dim=1)

#   return noise, np.squeeze(idx)

def controlled_noise_sample(batch_size, dim_discrete, **kwargs):
  # noise vector
  if('z_random' in kwargs.keys()):
    z_random = kwargs['z_random']
  else:
    z_random = torch.randn((batch_size, kwargs['nz'], 1, 1))

  # random categorical variable
  if('z_categorical' in kwargs.keys()):
    idx = kwargs['z_categorical']
  else:
    idx = torch.LongTensor(
      np.random.randint(
        dim_discrete, 
        size=(batch_size, kwargs['num_discrete'])), 
        )
  z_categorical = torch.zeros((batch_size, dim_discrete))
  z_categorical.scatter_(1, idx, 1.0)
  z_categorical = z_categorical.view(batch_size, -1, 1, 1)

  # random continuous variable (uniform between -1 and 1)
  if('z_continuous' in kwargs.keys()):
    z_continuous = kwargs['z_continuous']
  else:
    z_continuous = torch.rand((batch_size, kwargs['num_continuous'], 1, 1)) * 2 - 1  

  noise = torch.cat((z_random, z_categorical, z_continuous), dim=1)
  import pdb
  # pdb.set_trace()
  return noise, np.squeeze(idx)

def noise_sample(n_dis_c, dis_c_dim, n_con_c, nz, batch_size):
  """
  Sample random noise vector for training.
  
  Taken from: https://github.com/Natsu6767/InfoGAN-PyTorch

  INPUT
  --------
  n_dis_c : Number of discrete latent code.
  dis_c_dim : Dimension of discrete latent code.
  n_con_c : Number of continuous latent code.
  n_z : Dimension of iicompressible noise.
  batch_size : Batch Size
  """
  # noise vector
  z = torch.randn((batch_size, nz, 1, 1))

  # random categorical variable
  idx = np.zeros((batch_size, n_dis_c))
  if(n_dis_c != 0):
    dis_c = torch.zeros((batch_size, n_dis_c, dis_c_dim))
    
    for i in range(n_dis_c):
        idx[:, i] = np.random.randint(dis_c_dim, size=batch_size)
        dis_c[torch.arange(0, batch_size), i, idx[:, i]] = 1.0

    dis_c = dis_c.view(batch_size, -1, 1, 1)

  if(n_con_c != 0):
    # random uniform between -1 and 1  
    con_c = torch.rand((batch_size, n_con_c, 1, 1)) * 2 - 1
  noise = torch.cat((z, dis_c, con_c), dim=1)

  return noise, np.squeeze(idx)

class NormalNLLLoss():
  """
  Calculate the negative log likelihood of normal distribution.
  This needs to be minimised. Treating Q(cj | x) as a factored Gaussian.

  Taken from: https://github.com/Natsu6767/InfoGAN-PyTorch

  """
  def __init__(self):
    self.eps = 1e-8

  def __call__(self, x, mu, var):
    loglikelihood = -0.5 * torch.log(2 * np.pi * var) - (x - mu)**2 / (2 * var + self.eps)
    nll = - torch.sum(loglikelihood, dim=1).mean()

    return nll