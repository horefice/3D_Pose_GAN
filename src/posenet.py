import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""MyNN"""
class MyNN(nn.Module):
  """
  A PyTorch implementation of a superclass network.
  """

  def __init__(self):
    """
    Initialize a new network.
    """
    super(MyNN, self).__init__()

  def forward(self, x):
    """
    Forward pass of the neural network. Should not be called manually but by
    calling a model instance directly.

    Inputs:
    - x: PyTorch input Tensor
    """
    print("MyNN: Forward method should be overwritten!")
    return x

  @property
  def is_cuda(self):
    """
    Check if model parameters are allocated on the GPU.
    """
    return next(self.parameters()).is_cuda

"""PoseNet"""
class PoseNet(MyNN):
  """
  A PyTorch implementation of PoseNet
  """

  def __init__(self, n_in=34, n_hidden=1024, mode='generator'):
    if n_in % 2 != 0:
      raise ValueError("'n_in' must be divisible by 2.")
    if not mode in ['generator', 'discriminator']:
      raise ValueError("only 'generator' and 'discriminator' are valid "
                       "for 'mode', but '{}' is given.".format(mode))
    super(PoseNet, self).__init__()

    self.mode = mode
    n_out = n_in // 2 if mode == 'generator' else 1
    print('Model: {}, n_out: {}, n_hidden: {}'.format(mode, n_out, n_hidden))

    self.l1 = nn.Linear(n_in, n_hidden)
    self.l2 = nn.Linear(n_hidden, n_hidden)
    self.l3 = nn.Linear(n_hidden, n_hidden)
    self.l4 = nn.Linear(n_hidden, n_out)

  def forward(self, x):
    h1 = F.leaky_relu(self.l1(x))
    h2 = F.leaky_relu(self.l2(h1))
    h3 = F.leaky_relu(self.l3(h2) + h1)
    h4 = self.l4(h3)

    return h4 if self.mode == 'generator' else F.sigmoid(h4)

  def load_npz(self, path):
    npzfile = np.load(path)
    self.l1.weight = nn.Parameter(torch.from_numpy(npzfile['l1/W']))
    self.l1.bias = nn.Parameter(torch.from_numpy(npzfile['l1/b']))
    self.l2.weight = nn.Parameter(torch.from_numpy(npzfile['l2/W']))
    self.l2.bias = nn.Parameter(torch.from_numpy(npzfile['l2/b']))
    self.l3.weight = nn.Parameter(torch.from_numpy(npzfile['l3/W']))
    self.l3.bias = nn.Parameter(torch.from_numpy(npzfile['l3/b']))
    self.l4.weight = nn.Parameter(torch.from_numpy(npzfile['l4/W']))
    self.l4.bias = nn.Parameter(torch.from_numpy(npzfile['l4/b']))
