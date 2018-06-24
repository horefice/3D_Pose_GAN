import numpy as np
import cv2 as cv
import argparse
import torch
import os

from posenet import PoseNet
from dataHandler import MPII
import utils

## SETTINGS
parser = argparse.ArgumentParser(description='MyNet Implementation')
parser.add_argument('-x', '--expID', type=str, default='test', metavar='S',
                    help='Experiment ID')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', '--learning-rate', type=float, default=2e-4, metavar='F',
                    help='learning rate (default: 2e-4)')
parser.add_argument('--model', type=str, default='', metavar='S',
                    help='path to saved model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before saving (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging (default: 10)')

## SETUP
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.saveDir = os.path.join('../models/', args.expID)

device = torch.device("cpu")
torch.manual_seed(args.seed)
kwargs = {}
if args.cuda:
  device = torch.device("cuda")
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = True
  kwargs = {'num_workers': 1, 'pin_memory': True}

os.makedirs(args.saveDir, exist_ok=True)
utils.writeArgsFile(args,args.saveDir)

## LOAD DATASETS
print('\nDATASET INFO.')
train_data = MPII('../data/mpii_poses.npy')
print('Train size: {} x {}'.format(len(train_data), train_data[0].size()))

## LOAD MODEL
print('\nLOADING GAN.')

def weights_init(m):
    if type(m) == torch.nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      torch.nn.init.constant_(m.bias, 0.0)

netG = PoseNet(mode="generator").to(device)
netG.apply(weights_init)
netD = PoseNet(mode="discriminator").to(device)
netD.apply(weights_init)
if args.model:
  netG.load_state_dict(torch.load(args.model)['netG'])
  netD.load_state_dict(torch.load(args.model)['netD'])
  print("=> Loaded models from {:s}".format(args.model))
print("Model params: {:.2f}M".format(sum(p.numel() for p in netG.parameters()) / 1e6))

## TRAINING
print('\nTRAINING.')
data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          **kwargs)

# fixed_noise = torch.randn(1, 34, device=device)
one = torch.FloatTensor([1])
mone = one * -1

optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr)
optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr)

iter_per_epoch = len(data_loader)
start_epoch = torch.load(args.model)['epoch'] if args.model else 0

print('Start')
for epoch in range(start_epoch, args.epochs):
  for i, data in enumerate(data_loader, 1):
    batch_size = data.size(0)
    real_data = data.to(device)
    angle = 10/180*np.pi # min and (2pi-max) angle for rotation
    ############################
    # (1) Update G network: maximize E[D(G(x))]
    ###########################
    netG.zero_grad()
    for p in netD.parameters():
      p.requires_grad = False  # disable to avoid computation
    theta = angle + 2*(np.pi-angle)*torch.rand(batch_size, 17, device=device)
    
    z_pred = netG(real_data)
    fake_data = utils.rotate_and_project(real_data, z_pred, theta)

    G = netD(fake_data)
    G = G.mean()
    G.backward(mone)

    errG = -G
    optimizerG.step()

    ############################
    # (2) Update D network: maximize E[D(x)] - E[D(G(x))] + GP
    ###########################
    # train with real
    netD.zero_grad()
    for p in netD.parameters():
      p.requires_grad = True # enable back grad (see below)

    D_real = netD(real_data)
    D_real = D_real.mean()
    D_real.backward(mone)

    # train with fake
    # noise = torch.randn(batch_size, 34, device=device)
    theta = angle + 2*(np.pi-angle)*torch.rand(batch_size, 17, device=device)

    with torch.no_grad():
      z_pred = netG(real_data)
    fake_data = utils.rotate_and_project(real_data, z_pred, theta)

    D_fake = netD(fake_data)
    D_fake = D_fake.mean()
    D_fake.backward(one)

    # gradient penalty
    lambda_gp = 10
    GP = utils.calc_gradient_penalty(netD, real_data, fake_data,
                                     LAMBDA=lambda_gp, device=device)
    GP.backward()

    errD = D_fake - D_real + GP
    WD = D_real - D_fake
    optimizerD.step()

    # Log
    if args.log_interval and i % args.log_interval == 0:
      print(('[{:3d}/{:3d}][{:4d}/{:4d}] Loss_D: {:.4f} Loss_G: {:.4f} ' +
            'D(x): {:.4f} D(G(z)): {:.4f}')
            .format(epoch+1, args.epochs, i, iter_per_epoch,
                    errD.item(), errG.item(), D_real, D_fake))

  # do checkpointing
  if args.save_interval and (epoch+1) % args.save_interval == 0:
    torch.save({'epoch': epoch+1,
               'netG': netG.state_dict(),
               'netD': netD.state_dict()
               }, '{:s}/checkpoint.pth'.format(args.saveDir))
    print('Saved at checkpoint!')
print('Finish')
