import os

import torch
import torchvision.transforms as transforms

class MNIST_Dataset(torch.utils.data.Dataset):
  def __init__(self, dataroot, istrain=True, transform=None):
    # Configure data loader
    if(not os.path.exists(dataroot)):
      os.makedirs(dataroot, exist_ok=True)

    if(istrain):
      self.data = torch.load(os.path.join(dataroot, 'MNIST/processed/training.pt'))[0]
    else:
      self.data = torch.load(os.path.join(dataroot, 'MNIST/processed/test.pt'))[0]
    self.transform = transforms.Compose(transform)

  def __len__(self):
      return self.data.size(0)

  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      # sample
      image = 2 * (self.data[idx].type(torch.float32) / 255) - 1 # normalize image to range [-1, 1]
      if self.transform:
          image = self.transform(image)

      return image


class FashionMNIST_Dataset(torch.utils.data.Dataset):
  def __init__(self, dataroot, istrain=True, transform=None):
    # configure data loader
    if(not os.path.exists(dataroot)):
      os.makedirs(dataroot, exist_ok=True)

    if(istrain):
      self.data = torch.load(os.path.join(dataroot, 'FashionMNIST/processed/training.pt'))[0]
    else:
      self.data = torch.load(os.path.join(dataroot, 'FashionMNIST/processed/test.pt'))[0]
    self.transform = transforms.Compose(transform)

  def __len__(self):
      return self.data.size(0)

  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      # sample
      image = 2 * (self.data[idx].type(torch.float32) / 255) - 1 # normalize image to range [-1, 1]
      if self.transform:
          image = self.transform(image)

      return image
