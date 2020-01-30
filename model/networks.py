# author: Deepankar C.

import torch
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, in_channels, out_channels, cnum):
    super(Generator, self).__init__()
    self.model = nn.Sequential(
        nn.ConvTranspose2d(in_channels, cnum*16, 1, 1, bias=False), 
        nn.BatchNorm2d(cnum*16), nn.ReLU(True), 
        nn.ConvTranspose2d(cnum*16, cnum*2, 7, 1, bias=False), 
        nn.BatchNorm2d(cnum*2), nn.ReLU(True), 
        nn.ConvTranspose2d(cnum*2, cnum, 4, 2, padding=1, bias=False), 
        nn.BatchNorm2d(cnum), nn.ReLU(True), 
        nn.ConvTranspose2d(cnum, 1, 4, 2, padding=1, bias=False), 
        nn.Tanh()
        )

  def forward(self, input):
    return self.model(input)

class DisConLayers(nn.Module):
  def __init__(self, in_channels, out_channels, cnum):
    super(DisConLayers, self).__init__()
    self.model = nn.Sequential(
        nn.Conv2d(in_channels, cnum, 4, 2, 1), 
        nn.LeakyReLU(0.2, True), 
        nn.Conv2d(cnum, cnum*2, 4, 2, 1, bias=False), 
        nn.BatchNorm2d(cnum*2), nn.LeakyReLU(0.2, True), 
        nn.Conv2d(cnum*2, cnum*16, 7, bias=False), 
        nn.BatchNorm2d(cnum*16), nn.LeakyReLU(0.2, True)
        )

  def forward(self, input):
    return self.model(input)

class Discriminator(nn.Module):
  def __init__(self, cnum):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(nn.Conv2d(cnum*16, 1, 1, bias=False), nn.Sigmoid())

  def forward(self, input):
    return self.model(input)

class AuxDistribution(nn.Module):
  def __init__(self, cnum, ndc, ncc):
    super(AuxDistribution, self).__init__()
    self.model = nn.Sequential(
        nn.Conv2d(cnum*16, cnum*2, 1, bias=False), 
        nn.BatchNorm2d(cnum*2), nn.LeakyReLU(0.2, True)
        )
    self.con_categorical = nn.Conv2d(cnum*2, ndc, 1)
    self.con_mu = nn.Conv2d(cnum*2, ncc, 1)
    self.con_var = nn.Conv2d(cnum*2, ncc, 1)

  def forward(self, input):
    x = self.model(input)

    categorical_logits = self.con_categorical(x).squeeze()
    continuous_mu = self.con_mu(x).squeeze()
    continuous_var = torch.exp(self.con_var(x)).squeeze()

    return categorical_logits, continuous_mu, continuous_var

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)